{ ************************************************************************** }
{                                                                            }
{ CrossPascal.TVMCompiler                                                    }
{                                                                            }
{ Copyright © 2026 Yuriy Pisarev (ypisareff@outlook.com)                     }
{                                                                            }
{ ************************************************************************** }

unit CrossPascal.TVMCompiler;

{$IFDEF FPC}{$MODE DELPHIUNICODE}{$ENDIF}

interface

uses
  TVM;

type
  TWindowsTvmKit = record
    CompilerDirectory: string;
    Linker: string;
    FfiLibrary: string;
    MsvcLibraries: string;
    UcrtLibraries: string;
    WindowsLibraries: string;
  end;

procedure InitializeTvmCompiler(const Directory: string);
function CompilerObject(const Name: string; const Args: array of TTVMFFIAny): TTVMFFIObjectHandle;
procedure CompileCpuRelaxModule(const IR: TTVMFFIObjectHandle;
  const TargetJSON, KernelObject, VMObject: string);
procedure CompileCudaRelaxModule(const IR: TTVMFFIObjectHandle;
  const CompilerDirectory, NvrtcLibrary, Architecture, HostTargetJSON, KernelObject, VMObject: string);
procedure CompileCpuTIR(const CompilerDirectory, InputFile, TargetJSON, ObjectFile: string);
procedure CompileCpuRelax(const CompilerDirectory, InputFile, TargetJSON, KernelObject, VMObject: string);
procedure CompileCudaRelax(const CompilerDirectory, NvrtcLibrary, Architecture, InputFile, HostTargetJSON, KernelObject, VMObject: string);
procedure CompileCpuDense(const CompilerDirectory, TargetJSON, KernelObject, VMObject: string;
  const Rows, Columns, Features: Integer);
procedure LinkWindowsTvmLibrary(const Linker, OutputFile: string;
  const Objects, Libraries, LibraryDirectories: array of string);

implementation

uses
  {$IFDEF FPC}SysUtils, Classes, Windows{$ELSE}System.SysUtils, System.Classes, Winapi.Windows{$ENDIF},
  DLPack, TVM.CpuGraph, QwenCuda, CrossPascal.Nvrtc;

const
  CompilerName = 'tvm_compiler.dll';

var
  CompilerHandle: HMODULE;
  LoadedCompiler: string;

function GlobalObject(const Name: string; const Args: array of TTVMFFIAny): TTVMFFIObjectHandle;
var
  Value: TTVMFFIAny;
  Error: TTVMCallError;
begin
  if not TTVMRuntime.CallGlobal(Name, Args, Value, @Error) then
    raise Exception.Create(Name + ': ' + string(Error));
  RequireCpu((Value.TypeIndex >= kTVMFFIStaticObjectBegin) and (Value.VObj <> nil),
    Name + ' did not return a TVM object');
  Result := Value.VObj;
end;

function CompilerObject(const Name: string; const Args: array of TTVMFFIAny): TTVMFFIObjectHandle;
begin
  Result := GlobalObject(Name, Args);
end;

procedure LoadCompiler(const Directory: string);
var
  FileName: string;
begin
  FileName := Directory + CompilerName;
  if CompilerHandle <> 0 then
  begin
    RequireCpu(SameText(FileName, LoadedCompiler), 'Restart before selecting a different TVM compiler');
    Exit;
  end;
  InitializeRuntime(Directory, False);
  RequireCpu(GetModuleHandleW(PWideChar(CompilerName)) = 0,
    'The TVM compiler was already loaded from outside the selected directory');
  CompilerHandle := LoadLibraryExW(PWideChar(FileName), 0,
    LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR or LOAD_LIBRARY_SEARCH_SYSTEM32);
  if CompilerHandle = 0 then RaiseLastOSError;
  LoadedCompiler := FileName;
  { Registered compiler functions remain in the TVM FFI registry for this process. }
end;

procedure InitializeTvmCompiler(const Directory: string);
begin
  LoadCompiler(IncludeTrailingPathDelimiter(ExpandFileName(Directory)));
end;

function ReadIR(const FileName: string): UTF8String;
var
  Stream: TFileStream;
begin
  Stream := TFileStream.Create(FileName, fmOpenRead or fmShareDenyWrite);
  try
    RequireCpu((Stream.Size > 0) and (Stream.Size <= High(Integer)), 'Invalid TVM IR file size');
    SetLength(Result, Integer(Stream.Size));
    Stream.ReadBuffer(Result[1], Length(Result));
    RequireCpu(Pos(#0, Result) = 0, 'The TVM IR must be a UTF-8 JSON document');
  finally
    Stream.Free;
  end;
end;

procedure ApplyPass(var IR: TTVMFFIObjectHandle; const Name: string; const Args: array of TTVMFFIAny);
var
  Pass, Next: TTVMFFIObjectHandle;
begin
  Pass := GlobalObject(Name, Args);
  try
    try
      Next := GlobalObject('transform.RunPass', [TTVMRuntime.AnyObject(Pass), TTVMRuntime.AnyObject(IR)]);
    except
      on E: Exception do raise Exception.Create(Name + ': ' + E.Message);
    end;
    TTVMRuntime.ObjectRelease(IR);
    IR := Next;
  finally
    TTVMRuntime.ObjectRelease(Pass);
  end;
end;

procedure LowerCpuTIR(var IR: TTVMFFIObjectHandle; const Target: TTVMFFIObjectHandle);
begin
  ApplyPass(IR, 'tirx.transform.BindTarget', [TTVMRuntime.AnyObject(Target)]);
  ApplyPass(IR, 's_tir.transform.CanonicalizeLoop', []);
  ApplyPass(IR, 's_tir.transform.LowerInitBlock', []);
  ApplyPass(IR, 's_tir.transform.PlanAndUpdateBufferAllocationLocation', []);
  ApplyPass(IR, 's_tir.transform.ConvertBlocksToOpaque', []);
  ApplyPass(IR, 's_tir.transform.CompactBufferAllocation', [TTVMRuntime.AnyBool(True)]);
  ApplyPass(IR, 's_tir.transform.LowerMatchBuffer', []);
  ApplyPass(IR, 'tirx.transform.StmtSimplify', []);
  ApplyPass(IR, 's_tir.transform.LowerOpaqueBlock', []);
  ApplyPass(IR, 'tirx.transform.FlattenBuffer', []);
  ApplyPass(IR, 'tirx.transform.BF16ComputeLegalize', []);
  ApplyPass(IR, 'tirx.transform.NarrowDataType', [TTVMRuntime.AnyInt(32)]);
  ApplyPass(IR, 's_tir.transform.LoopPartition', []);
  ApplyPass(IR, 'tirx.transform.VectorizeLoop', [TTVMRuntime.AnyBool(True)]);
  ApplyPass(IR, 'tirx.transform.StorageRewrite', []);
  ApplyPass(IR, 'tirx.transform.UnrollLoop', []);
  ApplyPass(IR, 'tirx.transform.StmtSimplify', []);
  ApplyPass(IR, 'tirx.transform.RemoveNoOp', []);
  ApplyPass(IR, 'tirx.transform.CommonSubexprElim', []);
  ApplyPass(IR, 'tirx.transform.FP8ComputeLegalize', [TTVMRuntime.AnyCStr('float32')]);
  ApplyPass(IR, 'tirx.transform.VerifyMemory', []);
  ApplyPass(IR, 'tirx.transform.AnnotateEntryFunc', []);
  ApplyPass(IR, 'tirx.transform.SplitHostDevice', []);
  ApplyPass(IR, 'tirx.transform.MakePackedAPI', []);
  ApplyPass(IR, 'tirx.transform.FP8StorageLegalize', []);
  ApplyPass(IR, 'tirx.transform.BF16StorageLegalize', []);
  ApplyPass(IR, 'tirx.transform.LowerTVMBuiltin', []);
  ApplyPass(IR, 'tirx.transform.LowerIntrin', []);
end;

procedure WriteObject(const Module: TTVMFFIObjectHandle; const FileName: string);
var
  Destination: UTF8String;
  Value: TTVMFFIAny;
  Error: TTVMCallError;
begin
  Destination := UTF8Encode(ExpandFileName(FileName));
  RequireTVM(
    TTVMRuntime.CallGlobal(
      'ffi.ModuleWriteToFile',
      [
        TTVMRuntime.AnyObject(Module),
        TTVMRuntime.AnyCStr(PAnsiChar(Destination)),
        TTVMRuntime.AnyCStr('obj')
      ],
      Value,
      @Error
    ),
    @Error
  );
end;

function BuildCpuTIR(var IR: TTVMFFIObjectHandle; const Target: TTVMFFIObjectHandle): TTVMFFIObjectHandle;
var
  BoundTarget: TTVMFFIObjectHandle;
begin
  BoundTarget := GlobalObject('target.WithHost',
    [TTVMRuntime.AnyObject(Target), TTVMRuntime.AnyObject(Target)]);
  try
    LowerCpuTIR(IR, BoundTarget);
    Result := GlobalObject('target.build.llvm', [TTVMRuntime.AnyObject(IR), TTVMRuntime.AnyObject(Target)]);
  finally
    TTVMRuntime.ObjectRelease(BoundTarget);
  end;
end;

procedure LowerCudaTIR(var IR: TTVMFFIObjectHandle; const Target: TTVMFFIObjectHandle);
begin
  ApplyPass(IR, 'tirx.transform.BindTarget', [TTVMRuntime.AnyObject(Target)]);
  ApplyPass(IR, 's_tir.transform.CanonicalizeLoop', []);
  ApplyPass(IR, 's_tir.transform.LowerCrossThreadReduction', []);
  ApplyPass(IR, 's_tir.transform.LowerInitBlock', []);
  ApplyPass(IR, 's_tir.transform.PlanAndUpdateBufferAllocationLocation', []);
  ApplyPass(IR, 's_tir.transform.ConvertBlocksToOpaque', []);
  ApplyPass(IR, 's_tir.transform.LiftThreadBinding', []);
  ApplyPass(IR, 's_tir.transform.ManifestSharedMemoryLocalStage', []);
  ApplyPass(IR, 's_tir.transform.CompactBufferAllocation', [TTVMRuntime.AnyBool(True)]);
  ApplyPass(IR, 's_tir.transform.LowerAutoCopy', []);
  ApplyPass(IR, 's_tir.transform.UnifyThreadBinding', []);
  ApplyPass(IR, 's_tir.transform.LowerMatchBuffer', []);
  ApplyPass(IR, 'tirx.transform.StmtSimplify', []);
  ApplyPass(IR, 's_tir.transform.InjectPermutedLayout', []);
  ApplyPass(IR, 's_tir.transform.AnnotateIrregularLoop', []);
  ApplyPass(IR, 's_tir.transform.InjectSoftwarePipeline', []);
  ApplyPass(IR, 's_tir.transform.TransformMmaBufferLayout', []);
  ApplyPass(IR, 's_tir.transform.LowerOpaqueBlock', []);
  ApplyPass(IR, 'tirx.transform.FlattenBuffer', []);
  ApplyPass(IR, 'tirx.transform.BF16ComputeLegalize', []);
  ApplyPass(IR, 'tirx.transform.NarrowDataType', [TTVMRuntime.AnyInt(32)]);
  ApplyPass(IR, 's_tir.transform.LoopPartition', []);
  ApplyPass(IR, 'tirx.transform.VectorizeLoop', [TTVMRuntime.AnyBool(True)]);
  ApplyPass(IR, 's_tir.transform.InjectVirtualThread', []);
  ApplyPass(IR, 's_tir.transform.InjectDoubleBuffer', []);
  ApplyPass(IR, 'tirx.transform.StorageRewrite', []);
  ApplyPass(IR, 's_tir.transform.HoistIfThenElse', []);
  ApplyPass(IR, 'tirx.transform.UnrollLoop', []);
  ApplyPass(IR, 's_tir.transform.RenormalizeSplitPattern', []);
  ApplyPass(IR, 'tirx.transform.StmtSimplify', []);
  ApplyPass(IR, 'tirx.transform.RemoveNoOp', []);
  ApplyPass(IR, 's_tir.transform.RewriteUnsafeSelect', []);
  ApplyPass(IR, 'tirx.transform.CommonSubexprElim', []);
  ApplyPass(IR, 'tirx.transform.FP8ComputeLegalize', [TTVMRuntime.AnyCStr('float32')]);
  ApplyPass(IR, 'tirx.transform.VerifyMemory', []);
  ApplyPass(IR, 'tirx.transform.AnnotateEntryFunc', []);
  ApplyPass(IR, 's_tir.transform.ThreadSync', [TTVMRuntime.AnyCStr('shared')]);
  ApplyPass(IR, 's_tir.transform.ThreadSync', [TTVMRuntime.AnyCStr('shared.dyn')]);
  ApplyPass(IR, 's_tir.transform.ThreadSync', [TTVMRuntime.AnyCStr('warp')]);
  ApplyPass(IR, 's_tir.transform.InferFragment', []);
  ApplyPass(IR, 's_tir.transform.LowerThreadAllreduce', []);
  ApplyPass(IR, 's_tir.transform.MergeSharedMemoryAllocations', []);
  ApplyPass(IR, 'tirx.transform.SplitHostDevice', []);
  ApplyPass(IR, 'tirx.transform.MakePackedAPI', []);
  ApplyPass(IR, 'tirx.transform.FP8StorageLegalize', []);
  ApplyPass(IR, 'tirx.transform.BF16StorageLegalize', []);
end;

function GlobalInteger(const Name: string; const Args: array of TTVMFFIAny): Int64;
var
  Value: TTVMFFIAny;
  Error: TTVMCallError;
begin
  RequireTVM(TTVMRuntime.CallGlobal(Name, Args, Value, @Error), @Error);
  RequireCpu(Value.TypeIndex = kTVMFFIInt, 'Expected an integer from ' + Name);
  Result := Value.VInt64;
end;

function FunctionDevice(const Func: TTVMFFIObjectHandle): Int64;
var
  Attrs, Dict, Target: TTVMFFIObjectHandle;
begin
  Attrs := GlobalObject('ir.BaseFunc_Attrs', [TTVMRuntime.AnyObject(Func)]);
  try
    Dict := GlobalObject('ir.DictAttrsGetDict', [TTVMRuntime.AnyObject(Attrs)]);
    try
      Target := GlobalObject('ffi.MapGetItem', [TTVMRuntime.AnyObject(Dict), TTVMRuntime.AnyCStr('target')]);
      try
        Result := GlobalInteger('target.TargetGetDeviceType', [TTVMRuntime.AnyObject(Target)]);
      finally
        TTVMRuntime.ObjectRelease(Target);
      end;
    finally
      TTVMRuntime.ObjectRelease(Dict);
    end;
  finally
    TTVMRuntime.ObjectRelease(Attrs);
  end;
end;

procedure SplitCudaTIR(const IR: TTVMFFIObjectHandle; out Host, Device: TTVMFFIObjectHandle);
var
  Empty, Globals, Attrs, Key, Func: TTVMFFIObjectHandle;
  I, Count: Int64;

  procedure Add(var Module: TTVMFFIObjectHandle);
  var
    Next: TTVMFFIObjectHandle;
  begin
    Next := GlobalObject('ir.Module_Add',
      [TTVMRuntime.AnyObject(Module), TTVMRuntime.AnyObject(Key), TTVMRuntime.AnyObject(Func), TTVMRuntime.AnyBool(False)]);
    TTVMRuntime.ObjectRelease(Module);
    Module := Next;
  end;

begin
  Host := nil;
  Device := nil;
  Empty := GlobalObject('ffi.Map', []);
  try
    Attrs := GlobalObject('ir.Module_GetAttrs', [TTVMRuntime.AnyObject(IR)]);
    try
      Host := GlobalObject('ir.IRModule',
        [TTVMRuntime.AnyObject(Empty), TTVMRuntime.AnyObject(Attrs), TTVMRuntime.AnyObject(Empty)]);
      Device := GlobalObject('ir.IRModule',
        [TTVMRuntime.AnyObject(Empty), TTVMRuntime.AnyObject(Attrs), TTVMRuntime.AnyObject(Empty)]);
    finally
      TTVMRuntime.ObjectRelease(Attrs);
    end;
  finally
    TTVMRuntime.ObjectRelease(Empty);
  end;
  Globals := GlobalObject('ir.Module_GetGlobalVars', [TTVMRuntime.AnyObject(IR)]);
  try
    Count := GlobalInteger('ffi.ArraySize', [TTVMRuntime.AnyObject(Globals)]);
    for I := 0 to Count - 1 do
    begin
      Key := GlobalObject('ffi.ArrayGetItem', [TTVMRuntime.AnyObject(Globals), TTVMRuntime.AnyInt(I)]);
      try
        Func := GlobalObject('ir.Module_Lookup', [TTVMRuntime.AnyObject(IR), TTVMRuntime.AnyObject(Key)]);
        try
          case FunctionDevice(Func) of
            Ord(kDLCPU): Add(Host);
            Ord(kDLCUDA): Add(Device);
          else
            raise ECpuGraph.Create('Unexpected device in the lowered CUDA graph');
          end;
        finally
          TTVMRuntime.ObjectRelease(Func);
        end;
      finally
        TTVMRuntime.ObjectRelease(Key);
      end;
    end;
  finally
    TTVMRuntime.ObjectRelease(Globals);
  end;
end;

function BuildCudaTIR(var IR: TTVMFFIObjectHandle; const HostTarget, CudaTarget: TTVMFFIObjectHandle;
  const Compiler: TNvrtcCompiler): TTVMFFIObjectHandle;
var
  Host, Device, GPU, Module: TTVMFFIObjectHandle;
  Error: TTVMCallError;
  Value: TTVMFFIAny;
begin
  Host := nil;
  Device := nil;
  GPU := nil;
  Module := nil;
  try
    LowerCudaTIR(IR, CudaTarget);
    SplitCudaTIR(IR, Host, Device);
    ApplyPass(Host, 'tirx.transform.LowerTVMBuiltin', []);
    ApplyPass(Host, 'tirx.transform.LowerIntrin', []);
    ApplyPass(Device, 'tirx.transform.LowerWarpMemory', []);
    ApplyPass(Device, 'tirx.transform.StmtSimplify', []);
    ApplyPass(Device, 'tirx.transform.LowerIntrin', []);
    GPU := Compiler.BuildDeviceModule(Device, CudaTarget);
    Module := GlobalObject('target.build.llvm',
      [TTVMRuntime.AnyObject(Host), TTVMRuntime.AnyObject(HostTarget)]);
    RequireTVM(TTVMRuntime.CallGlobal('ffi.ModuleImportModule',
      [TTVMRuntime.AnyObject(Module), TTVMRuntime.AnyObject(GPU)], Value, @Error), @Error);
    Result := Module;
    Module := nil;
  finally
    TTVMRuntime.ObjectRelease(Module);
    TTVMRuntime.ObjectRelease(GPU);
    TTVMRuntime.ObjectRelease(Device);
    TTVMRuntime.ObjectRelease(Host);
  end;
end;

function CreateCpuTarget(const TargetJSON: UTF8String): TTVMFFIObjectHandle;
var
  Value: TTVMFFIAny;
  Error: TTVMCallError;
begin
  Result := GlobalObject('target.Target', [TTVMRuntime.AnyCStr(PAnsiChar(TargetJSON))]);
  try
    RequireTVM(TTVMRuntime.CallGlobal('target.TargetGetDeviceType', [TTVMRuntime.AnyObject(Result)],
      Value, @Error), @Error);
    RequireCpu((Value.TypeIndex = kTVMFFIInt) and (Value.VInt64 = 1), 'An LLVM CPU target is required');
  except
    TTVMRuntime.ObjectRelease(Result);
    raise;
  end;
end;

function ImportIR(const InputFile: string): TTVMFFIObjectHandle;
var
  Source: UTF8String;
begin
  Source := ReadIR(InputFile);
  Result := GlobalObject('ffi.FromJSONGraphString', [TTVMRuntime.AnyCStr(PAnsiChar(Source))]);
end;

procedure CompileCpuTIR(const CompilerDirectory, InputFile, TargetJSON, ObjectFile: string);
var
  IR, Target, Code: TTVMFFIObjectHandle;
begin
  LoadCompiler(IncludeTrailingPathDelimiter(ExpandFileName(CompilerDirectory)));
  Target := CreateCpuTarget(UTF8Encode(TargetJSON));
  IR := nil;
  Code := nil;
  try
    IR := ImportIR(InputFile);
    Code := BuildCpuTIR(IR, Target);
    WriteObject(Code, ObjectFile);
  finally
    TTVMRuntime.ObjectRelease(Code);
    TTVMRuntime.ObjectRelease(IR);
    TTVMRuntime.ObjectRelease(Target);
  end;
end;

procedure EmitRelax(var IR: TTVMFFIObjectHandle; const TargetData: UTF8String;
  const KernelObject, VMObject: string; const Compiler: TNvrtcCompiler = nil;
  const CudaTargetData: UTF8String = '');
var
  Target, TIR, Code, Builder, Libraries, Constants, Executable, Imports: TTVMFFIObjectHandle;
  CudaTarget, BoundTarget, LinkTarget: TTVMFFIObjectHandle;
begin
  RequireCpu(not SameText(ExpandFileName(KernelObject), ExpandFileName(VMObject)),
    'Choose separate paths for the CPU kernels and the TVM VM executable');
  Target := CreateCpuTarget(TargetData);
  TIR := nil;
  Code := nil;
  Builder := nil;
  Libraries := nil;
  Constants := nil;
  Executable := nil;
  Imports := nil;
  CudaTarget := nil;
  BoundTarget := nil;
  LinkTarget := Target;
  try
    { Input operators must already be expressed as call_tir or TVM VM builtins. }
    ApplyPass(IR, 'relax.transform.RewriteDataflowReshape', []);
    ApplyPass(IR, 'relax.transform.ToNonDataflow', []);
    ApplyPass(IR, 'relax.transform.RemovePurityChecking', []);
    ApplyPass(IR, 'relax.transform.CallTIRRewrite', []);
    ApplyPass(IR, 'relax.transform.StaticPlanBlockMemory', []);
    ApplyPass(IR, 'relax.transform.LowerAllocTensor', []);
    ApplyPass(IR, 'relax.transform.KillAfterLastUse', []);
    ApplyPass(IR, 'relax.transform.LowerRuntimeBuiltin', []);
    ApplyPass(IR, 'relax.transform.ComputePrimValue', []);
    ApplyPass(IR, 'relax.transform.VMShapeLower', [TTVMRuntime.AnyBool(True)]);
    ApplyPass(IR, 'relax.transform.AttachGlobalSymbol', []);
    Builder := GlobalObject('relax.ExecBuilderCreate', []);
    TIR := GlobalObject('relax.VMCodeGen', [TTVMRuntime.AnyObject(Builder), TTVMRuntime.AnyObject(IR)]);
    if Compiler = nil then
      Code := BuildCpuTIR(TIR, Target)
    else begin
      CudaTarget := GlobalObject('target.Target', [TTVMRuntime.AnyCStr(PAnsiChar(CudaTargetData))]);
      BoundTarget := GlobalObject('target.WithHost',
        [TTVMRuntime.AnyObject(CudaTarget), TTVMRuntime.AnyObject(Target)]);
      Code := BuildCudaTIR(TIR, Target, BoundTarget, Compiler);
      LinkTarget := BoundTarget;
    end;
    Libraries := GlobalObject('ffi.Array', []);
    Constants := GlobalObject('ffi.Map', []);
    Executable := GlobalObject(
      'relax.VMLink',
      [
        TTVMRuntime.AnyObject(Builder),
        TTVMRuntime.AnyObject(LinkTarget),
        TTVMRuntime.AnyObject(Code),
        TTVMRuntime.AnyObject(Libraries),
        TTVMRuntime.AnyObject(Constants)
      ]
    );
    Imports := GlobalObject('runtime.ModulePackImportsToLLVM',
      [TTVMRuntime.AnyObject(Executable), TTVMRuntime.AnyBool(False), TTVMRuntime.AnyCStr(PAnsiChar(TargetData)), TTVMRuntime.AnyCStr('')]);
    WriteObject(Code, KernelObject);
    WriteObject(Imports, VMObject);
  finally
    TTVMRuntime.ObjectRelease(Imports);
    TTVMRuntime.ObjectRelease(Executable);
    TTVMRuntime.ObjectRelease(Constants);
    TTVMRuntime.ObjectRelease(Libraries);
    TTVMRuntime.ObjectRelease(Builder);
    TTVMRuntime.ObjectRelease(Code);
    TTVMRuntime.ObjectRelease(TIR);
    TTVMRuntime.ObjectRelease(BoundTarget);
    TTVMRuntime.ObjectRelease(CudaTarget);
    TTVMRuntime.ObjectRelease(Target);
  end;
end;

procedure CompileCpuRelax(const CompilerDirectory, InputFile, TargetJSON, KernelObject, VMObject: string);
var
  IR: TTVMFFIObjectHandle;
begin
  LoadCompiler(IncludeTrailingPathDelimiter(ExpandFileName(CompilerDirectory)));
  IR := ImportIR(InputFile);
  try
    EmitRelax(IR, UTF8Encode(TargetJSON), KernelObject, VMObject);
  finally
    TTVMRuntime.ObjectRelease(IR);
  end;
end;

procedure CompileCudaRelax(const CompilerDirectory, NvrtcLibrary, Architecture, InputFile, HostTargetJSON, KernelObject, VMObject: string);
var
  Directory: string;
  IR: TTVMFFIObjectHandle;
  Compiler: TNvrtcCompiler;
begin
  Directory := IncludeTrailingPathDelimiter(ExpandFileName(CompilerDirectory));
  LoadCompiler(Directory);
  InitializeRuntime(Directory, True);
  IR := nil;
  Compiler := TNvrtcCompiler.Create(NvrtcLibrary, Architecture);
  try
    IR := ImportIR(InputFile);
    EmitRelax(IR, UTF8Encode(HostTargetJSON), KernelObject, VMObject, Compiler,
      UTF8Encode('{"kind":"cuda","arch":"' + Architecture + '"}'));
  finally
    TTVMRuntime.ObjectRelease(IR);
    Compiler.Free;
  end;
end;

procedure CompileCpuRelaxModule(const IR: TTVMFFIObjectHandle;
  const TargetJSON, KernelObject, VMObject: string);
var
  Module: TTVMFFIObjectHandle;
begin
  RequireCpu(IR <> nil, 'A TVM Relax module is required');
  TTVMRuntime.ObjectRetain(IR);
  Module := IR;
  try
    ApplyPass(Module, 'relax.transform.Normalize', []);
    EmitRelax(Module, UTF8Encode(TargetJSON), KernelObject, VMObject);
  finally
    TTVMRuntime.ObjectRelease(Module);
  end;
end;

procedure CompileCudaRelaxModule(const IR: TTVMFFIObjectHandle;
  const CompilerDirectory, NvrtcLibrary, Architecture, HostTargetJSON, KernelObject, VMObject: string);
var
  Directory: string;
  Module: TTVMFFIObjectHandle;
  Compiler: TNvrtcCompiler;
  TargetData: UTF8String;
begin
  RequireCpu(IR <> nil, 'A TVM Relax module is required');
  Directory := IncludeTrailingPathDelimiter(ExpandFileName(CompilerDirectory));
  LoadCompiler(Directory);
  InitializeRuntime(Directory, True);
  Compiler := TNvrtcCompiler.Create(NvrtcLibrary, Architecture);
  TTVMRuntime.ObjectRetain(IR);
  Module := IR;
  try
    ApplyPass(Module, 'relax.transform.Normalize', []);
    TargetData := UTF8Encode('{"kind":"cuda","arch":"' + Architecture + '"}');
    EmitRelax(Module, UTF8Encode(HostTargetJSON), KernelObject, VMObject, Compiler, TargetData);
  finally
    TTVMRuntime.ObjectRelease(Module);
    Compiler.Free;
  end;
end;

function BuildDenseIR(const Rows, Columns, Features: Integer): TTVMFFIObjectHandle;
var
  Owned: TArray<TTVMFFIObjectHandle>;
  Empty, FP32, I64, TensorA, TensorB, ProductTensor, Kernel, Dense: TTVMFFIAny;
  InputShape, WeightShape, OutputShape, OutputType, InputA, InputB, Call, Func: TTVMFFIAny;
  EmptyMap, EmptyModule, Attributes, Builder: TTVMFFIAny;
  DType: TDLDataType;
  I: Integer;

  function Keep(const Name: string; const Args: array of TTVMFFIAny): TTVMFFIAny;
  var
    Slot: Integer;
  begin
    Slot := Length(Owned);
    SetLength(Owned, Slot + 1);
    Owned[Slot] := GlobalObject(Name, Args);
    Result := TTVMRuntime.AnyObject(Owned[Slot]);
  end;

  function Shape(const First, Second: Integer): TTVMFFIAny;
  var
    Dim1, Dim2: TTVMFFIAny;
  begin
    Dim1 := Keep('ir.IntImm', [I64, TTVMRuntime.AnyInt(First), Empty]);
    Dim2 := Keep('ir.IntImm', [I64, TTVMRuntime.AnyInt(Second), Empty]);
    Result := Keep('ffi.Array', [Dim1, Dim2]);
  end;

  function TensorType(const Dimensions: TTVMFFIAny): TTVMFFIAny;
  var
    ShapeExpr: TTVMFFIAny;
  begin
    ShapeExpr := Keep('relax.ShapeExpr', [Dimensions, Empty]);
    Result := Keep('relax.TensorType', [ShapeExpr, FP32, TTVMRuntime.AnyInt(-1), Empty, Empty]);
  end;

begin
  RequireCpu((Rows > 0) and (Columns > 0) and (Features > 0), 'Dense tensor dimensions must be positive');
  Owned := nil;
  Empty := Default(TTVMFFIAny);
  DType.lanes := 1;
  DType.code := Ord(kDLInt);
  DType.bits := 64;
  I64 := TTVMRuntime.AnyDataType(DType);
  DType.code := Ord(kDLFloat);
  DType.bits := 32;
  try
    FP32 := Keep('ir.PrimType', [TTVMRuntime.AnyDataType(DType)]);
    InputShape := Shape(Rows, Features);
    WeightShape := Shape(Columns, Features);
    OutputShape := Shape(Rows, Columns);
    TensorA := Keep('te.Placeholder', [InputShape, FP32, TTVMRuntime.AnyCStr('input')]);
    TensorB := Keep('te.Placeholder', [WeightShape, FP32, TTVMRuntime.AnyCStr('weights')]);
    ProductTensor := Keep('topi.nn.dense', [TensorA, TensorB, Empty, FP32]);
    Kernel := Keep('te.CreatePrimFunc', [Keep('ffi.Array', [TensorA, TensorB, ProductTensor]), Empty]);
    Kernel := Keep('ir.BaseFuncWithAttr',
      [Kernel, TTVMRuntime.AnyCStr('global_symbol'), TTVMRuntime.AnyCStr('dense')]);
    EmptyMap := Keep('ffi.Map', []);
    EmptyModule := Keep('ir.IRModule', [EmptyMap, EmptyMap, EmptyMap]);
    Attributes := Keep('ir.Module_GetAttrs', [EmptyModule]);
    Builder := Keep('relax.BlockBuilderCreate', [EmptyModule]);
    Dense := Keep('relax.BlockBuilderAddFunction', [Builder, Kernel, TTVMRuntime.AnyCStr('dense')]);
    InputA := Keep('ir.Var', [TTVMRuntime.AnyCStr('input'), TensorType(InputShape), Empty]);
    InputB := Keep('ir.Var', [TTVMRuntime.AnyCStr('weights'), TensorType(WeightShape), Empty]);
    OutputType := TensorType(OutputShape);
    Call := Keep('relax.op.call_tir',
      [Dense, Keep('relax.Tuple', [Keep('ffi.Array', [InputA, InputB]), Empty]), Keep('ffi.Array', [OutputType])]);
    Func := Keep('relax.Function', [Keep('ffi.Array', [InputA, InputB]), Call, OutputType,
      TTVMRuntime.AnyBool(True), Attributes, Empty]);
    Keep('relax.BlockBuilderAddFunction', [Builder, Func, TTVMRuntime.AnyCStr('main')]);
    Result := GlobalObject('relax.BlockBuilderFinalize', [Builder]);
  finally
    for I := High(Owned) downto 0 do TTVMRuntime.ObjectRelease(Owned[I]);
  end;
end;

procedure CompileCpuDense(const CompilerDirectory, TargetJSON, KernelObject, VMObject: string;
  const Rows, Columns, Features: Integer);
var
  IR: TTVMFFIObjectHandle;
begin
  LoadCompiler(IncludeTrailingPathDelimiter(ExpandFileName(CompilerDirectory)));
  IR := BuildDenseIR(Rows, Columns, Features);
  try
    ApplyPass(IR, 'relax.transform.Normalize', []);
    EmitRelax(IR, UTF8Encode(TargetJSON), KernelObject, VMObject);
  finally
    TTVMRuntime.ObjectRelease(IR);
  end;
end;

function LinkArgument(const Value: string): string;
begin
  RequireCpu((Value <> '') and (Pos('"', Value) = 0) and (Pos(#0, Value) = 0),
    'Invalid Windows linker argument');
  Result := ' "' + Value + '"';
end;

procedure LinkWindowsTvmLibrary(const Linker, OutputFile: string;
  const Objects, Libraries, LibraryDirectories: array of string);
var
  Tool, Command, Destination, Item: string;
  Startup: TStartupInfoW;
  Process: TProcessInformation;
  ExitCode: DWORD;
begin
  Tool := ExpandFileName(Linker);
  Destination := ExpandFileName(OutputFile);
  RequireCpu(FileExists(Tool), 'Windows linker not found: ' + Tool);
  RequireCpu(Length(Objects) > 0, 'A TVM library needs at least one object file');
  RequireCpu(DirectoryExists(ExtractFileDir(Destination)), 'Output directory not found');
  Command := Copy(LinkArgument(Tool), 2, MaxInt) + ' /NOLOGO /DLL /MACHINE:X64 /BREPRO' +
    LinkArgument('/OUT:' + Destination);
  for Item in LibraryDirectories do
    Command := Command +
      LinkArgument('/LIBPATH:' + ExcludeTrailingPathDelimiter(ExpandFileName(Item)));
  for Item in Objects do Command := Command + LinkArgument(ExpandFileName(Item));
  for Item in Libraries do Command := Command + LinkArgument(Item);
  FillChar(Startup, SizeOf(Startup), 0);
  Startup.cb := SizeOf(Startup);
  UniqueString(Command);
  if not CreateProcessW(PWideChar(Tool), PWideChar(Command), nil, nil, False, 0, nil, nil,
    Startup, Process) then RaiseLastOSError;
  try
    if WaitForSingleObject(Process.hProcess, INFINITE) <> WAIT_OBJECT_0 then RaiseLastOSError;
    if not GetExitCodeProcess(Process.hProcess, ExitCode) then RaiseLastOSError;
    RequireCpu(ExitCode = 0, Format('Windows linker exited with code %d', [ExitCode]));
  finally
    CloseHandle(Process.hThread);
    CloseHandle(Process.hProcess);
  end;
end;

end.
