<#
.SYNOPSIS
    Records the hardware and software environment used for the benchmarks.

.DESCRIPTION
    Writes results/environment.txt with: CPU model, physical cores, logical
    processors, per-instance L1/L2/L3 cache sizes, RAM, OS, compiler version,
    OpenMP version, Python and library versions, power plan and the
    threading-related environment variables in effect.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts/collect_env.ps1
#>
param(
    [string]$GccDir = "C:\msys64\ucrt64\bin",
    [string]$Python = "python",
    [string]$OutFile = ""
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
if (-not $OutFile) { $OutFile = Join-Path $RepoRoot "results\environment.txt" }
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $OutFile) | Out-Null

$lines = New-Object System.Collections.Generic.List[string]
function Add-Line([string]$text = "") { $lines.Add($text) }
function Add-Section([string]$title) { Add-Line ""; Add-Line "== $title =="; }

# Runs a native command and returns its trimmed output (stdout + stderr), or an
# error marker if the command is missing or exits with a non-zero code.
# ErrorActionPreference is relaxed locally because in Windows PowerShell 5.1 any
# stderr line from a native command would otherwise become a terminating error.
function Invoke-Safe([scriptblock]$block) {
    $prev = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        $global:LASTEXITCODE = 0
        $out = & $block 2>&1 | ForEach-Object { "$_" }
        $text = (($out -join "`n").Trim())
        if ($null -ne $LASTEXITCODE -and $LASTEXITCODE -ne 0) {
            return "UNAVAILABLE (exit code $LASTEXITCODE): $text"
        }
        return $text
    } catch {
        return "UNAVAILABLE ($($_.Exception.Message))"
    } finally {
        $ErrorActionPreference = $prev
    }
}

# Exact cache topology through the Win32 API (WMI only reports aggregated totals).
# If the type cannot be compiled, the cache section falls back to WMI.
$cacheTypeError = $null
if (-not ('CpuTopology' -as [type])) {
try {
Add-Type -TypeDefinition @"
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;

public static class CpuTopology {
    [StructLayout(LayoutKind.Sequential)]
    public struct CACHE_DESCRIPTOR {
        public byte Level;
        public byte Associativity;
        public ushort LineSize;
        public uint Size;
        public int Type; // 0 Unified, 1 Instruction, 2 Data, 3 Trace
    }

    [StructLayout(LayoutKind.Explicit)]
    public struct SLPI {
        [FieldOffset(0)]  public UIntPtr ProcessorMask;
        [FieldOffset(8)]  public int Relationship; // 0 Core, 1 NUMA, 2 Cache, 3 Package
        [FieldOffset(16)] public byte ProcessorCoreFlags;
        [FieldOffset(16)] public CACHE_DESCRIPTOR Cache;
        [FieldOffset(16)] public ulong Reserved0;
        [FieldOffset(24)] public ulong Reserved1;
    }

    [DllImport("kernel32.dll", SetLastError = true)]
    static extern bool GetLogicalProcessorInformation(IntPtr buffer, ref uint length);

    public static SLPI[] Query() {
        uint len = 0;
        GetLogicalProcessorInformation(IntPtr.Zero, ref len);
        IntPtr buf = Marshal.AllocHGlobal((int)len);
        try {
            if (!GetLogicalProcessorInformation(buf, ref len))
                throw new System.ComponentModel.Win32Exception(Marshal.GetLastWin32Error());
            int size = Marshal.SizeOf(typeof(SLPI));
            int count = (int)len / size;
            SLPI[] result = new SLPI[count];
            for (int i = 0; i < count; i++)
                result[i] = (SLPI)Marshal.PtrToStructure(new IntPtr(buf.ToInt64() + (long)i * size), typeof(SLPI));
            return result;
        } finally {
            Marshal.FreeHGlobal(buf);
        }
    }

    public static int PopCount(UIntPtr mask) {
        ulong v = mask.ToUInt64(); int c = 0;
        while (v != 0) { c += (int)(v & 1UL); v >>= 1; }
        return c;
    }
}
"@
} catch {
    $cacheTypeError = $_.Exception.Message
}
}

$timestamp = (Get-Date).ToString("yyyy-MM-ddTHH:mm:sszzz")
Add-Line "Environment report"
Add-Line "Generated: $timestamp"
Add-Line "Host: $env:COMPUTERNAME"
$gitRev = Invoke-Safe { git -C $RepoRoot rev-parse --short HEAD }
Add-Line "Git revision: $gitRev"

# ---------------------------------------------------------------- CPU
Add-Section "CPU"
$cpus = @(Get-CimInstance Win32_Processor)
$cpu = $cpus[0]
$physCores = ($cpus | Measure-Object -Property NumberOfCores -Sum).Sum
$logical = ($cpus | Measure-Object -Property NumberOfLogicalProcessors -Sum).Sum
Add-Line "Model: $($cpu.Name.Trim())"
Add-Line "Sockets: $($cpus.Count)"
Add-Line "Physical cores: $physCores"
Add-Line "Logical processors: $logical"
Add-Line "SMT (Hyper-Threading) active: $([bool]($logical -gt $physCores))"
Add-Line "Base clock (MHz): $($cpu.MaxClockSpeed)"
Add-Line "Current clock at collection (MHz): $($cpu.CurrentClockSpeed)"
Add-Line "Architecture: $env:PROCESSOR_ARCHITECTURE / $env:PROCESSOR_IDENTIFIER"

# ---------------------------------------------------------------- Caches
Add-Section "Caches (GetLogicalProcessorInformation)"
try {
    if ($cacheTypeError) { throw "CpuTopology type unavailable: $cacheTypeError" }
    # The struct offsets in CpuTopology assume the x64 layout (32-byte entries).
    if (-not [Environment]::Is64BitProcess) { throw "32-bit PowerShell is not supported for cache topology" }
    $info = [CpuTopology]::Query()
    $typeNames = @{ 0 = "Unified"; 1 = "Instruction"; 2 = "Data"; 3 = "Trace" }
    $caches = $info | Where-Object { $_.Relationship -eq 2 } | ForEach-Object {
        [pscustomobject]@{
            Level        = $_.Cache.Level
            Type         = $typeNames[[int]$_.Cache.Type]
            SizeKiB      = $_.Cache.Size / 1KB
            LineSize     = $_.Cache.LineSize
            Assoc        = $_.Cache.Associativity
            SharedByLP   = [CpuTopology]::PopCount($_.ProcessorMask)
        }
    }
    $caches | Group-Object Level, Type, SizeKiB, LineSize, Assoc, SharedByLP | Sort-Object { $_.Group[0].Level }, { $_.Group[0].Type } | ForEach-Object {
        $c = $_.Group[0]
        Add-Line ("L{0} {1,-11}: {2} KiB per instance x {3} instances (line {4} B, {5}-way, shared by {6} logical processors)" -f `
            $c.Level, $c.Type, $c.SizeKiB, $_.Count, $c.LineSize, $c.Assoc, $c.SharedByLP)
    }
    $coreCount = @($info | Where-Object { $_.Relationship -eq 0 }).Count
    Add-Line "Processor cores reported by API: $coreCount"
} catch {
    Add-Line "Cache query failed: $($_.Exception.Message)"
    Add-Line "WMI fallback: L2 total $($cpu.L2CacheSize) KiB, L3 total $($cpu.L3CacheSize) KiB"
}

# ---------------------------------------------------------------- Memory
Add-Section "Memory"
$cs = Get-CimInstance Win32_ComputerSystem
$os = Get-CimInstance Win32_OperatingSystem
# Invariant culture so numbers always use '.' as decimal separator.
$inv = [Globalization.CultureInfo]::InvariantCulture
Add-Line ([string]::Format($inv, "Total physical RAM: {0:F2} GiB ({1} bytes)", ($cs.TotalPhysicalMemory / 1GB), $cs.TotalPhysicalMemory))
Add-Line ([string]::Format($inv, "Free physical RAM at collection: {0:F2} GiB", ($os.FreePhysicalMemory * 1KB / 1GB)))
Get-CimInstance Win32_PhysicalMemory | ForEach-Object {
    Add-Line ([string]::Format($inv, "DIMM {0}: {1:F0} GiB @ {2} MT/s ({3})", $_.DeviceLocator, ($_.Capacity / 1GB), $_.ConfiguredClockSpeed, $_.Manufacturer))
}

# ---------------------------------------------------------------- OS
Add-Section "Operating system"
Add-Line "OS: $($os.Caption) $($os.Version) (build $($os.BuildNumber)), $($os.OSArchitecture)"
$powerPlan = Invoke-Safe { powercfg /getactivescheme }
Add-Line "Active power plan: $powerPlan"

# ---------------------------------------------------------------- Toolchain
Add-Section "Compiler"
$gcc = Join-Path $GccDir "gcc.exe"
if (Test-Path $gcc) {
    # gcc's helper programs (cc1, as, ld) need their DLLs from the same directory.
    $env:PATH = "$GccDir;$env:PATH"
    Add-Line "gcc path: $gcc"
    Add-Line ("gcc version: " + ((Invoke-Safe { & $gcc --version }) -split "`n")[0])
    Add-Line ("gcc target: " + (Invoke-Safe { & $gcc -dumpmachine }))
    $ompMacro = Invoke-Safe { "" | & $gcc -fopenmp -dM -E -x c - | Select-String "_OPENMP" }
    if (-not $ompMacro) { $ompMacro = "NOT DETECTED (gcc -fopenmp preprocessing failed)" }
    Add-Line "OpenMP macro: $ompMacro (201511 = OpenMP 4.5)"
    $objdump = Join-Path $GccDir "objdump.exe"
    if (Test-Path $objdump) { Add-Line ("objdump: " + ((Invoke-Safe { & $objdump --version }) -split "`n")[0]) }
} else {
    Add-Line "gcc NOT FOUND at $gcc"
}

# ---------------------------------------------------------------- Python
Add-Section "Python"
$pyScript = @"
import sys, importlib
print('Python:', sys.version.replace('\n', ' '))
for name in ('numpy', 'pandas', 'matplotlib', 'scipy'):
    try:
        m = importlib.import_module(name)
        print(f'{name}: {m.__version__}')
    except Exception as exc:
        print(f'{name}: NOT AVAILABLE ({exc})')
try:
    import numpy as np
    cfg = np.show_config(mode='dicts')
    blas = cfg.get('Build Dependencies', {}).get('blas', {})
    print('numpy BLAS:', blas.get('name'), blas.get('version'))
except Exception as exc:
    print('numpy BLAS: unknown', exc)
"@
Add-Line (Invoke-Safe { $pyScript | & $Python - })

# ---------------------------------------------------------------- Threading env
Add-Section "Threading-related environment variables (at collection time)"
foreach ($v in "OMP_NUM_THREADS", "OMP_DYNAMIC", "OMP_PROC_BIND", "OMP_PLACES", "OMP_WAIT_POLICY",
               "GOMP_SPINCOUNT", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS") {
    $val = [Environment]::GetEnvironmentVariable($v)
    if ($null -eq $val) { $val = "<unset>" }
    Add-Line "${v}=$val"
}
Add-Line "Note: the benchmark runner sets these explicitly per run and records them in the raw CSV."

$lines | Out-File -FilePath $OutFile -Encoding utf8
Write-Host "Environment written to $OutFile"
Get-Content $OutFile
