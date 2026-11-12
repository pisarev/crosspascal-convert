{ ************************************************************************** }
{                                                                            }
{ CrossPascal.Nvrtc                                                          }
{                                                                            }
{ Copyright © 2026 Yuriy Pisarev                                             }
{                                                                            }
{ ************************************************************************** }

unit CrossPascal.Nvrtc;

{$IFDEF FPC}{$MODE DELPHIUNICODE}{$ENDIF}
{$POINTERMATH ON}

interface

uses
  {$IFDEF FPC}SysUtils, Windows{$ELSE}System.SysUtils, Winapi.Windows{$ENDIF}, TVM;

type
  TNvrtcCompiler = class
  private type
    TCreate = function(out ProgramHandle: Pointer; Source, Name: PAnsiChar; Headers: Integer;
      HeaderSources, HeaderNames: PPAnsiChar): Integer; cdecl;
    TDestroy = function(var ProgramHandle: Pointer): Integer; cdecl;
    TCompile = function(ProgramHandle: Pointer; Options: Integer; Arguments: PPAnsiChar): Integer; cdecl;
    TGetSize = function(ProgramHandle: Pointer; out Size: NativeUInt): Integer; cdecl;
    TGetData = function(ProgramHandle: Pointer; Output: PAnsiChar): Integer; cdecl;
    TGetErrorString = function(Status: Integer): PAnsiChar; cdecl;
    TGetVersion = function(out Major, Minor: Integer): Integer; cdecl;
    TBytesFromArray = function(const Data: PTVMFFIByteArray; Output: PTVMFFIAny): Integer; cdecl;
  private
    FLibrary: HMODULE;
    FBuiltins: HMODULE;
    FArchitecture: UTF8String;
    FIncludePath: UTF8String;
    FCreate: TCreate;
    FDestroy: TDestroy;
    FCompile: TCompile;
    FCubinSize: TGetSize;
    FLogSize: TGetSize;
    FCubin: TGetData;
    FLog: TGetData;
    FError: TGetErrorString;
    FBytesFromArray: TBytesFromArray;
    FStringFromArray: TBytesFromArray;
    function Entry(const Name: AnsiString): Pointer;
    procedure Check(const Status: Integer; const Operation: string);
    function CompileSource(const Source: UTF8String): TBytes;
  public
    constructor Create(const LibraryFile, Architecture: string);
    destructor Destroy; override;
    function BuildDeviceModule(const IR, Target: TTVMFFIObjectHandle): TTVMFFIObjectHandle;
  end;

implementation

uses
  TVM.CpuGraph;

var
  CallbackRegistered, HeaderRegistered, IntrinsicsRegistered: Boolean;

threadvar
  ActiveCompiler: TNvrtcCompiler;

function SourceString(const Arg: PTVMFFIAny): UTF8String;
var
  ArrayView: PTVMFFIByteArray;
begin
  case Arg^.TypeIndex of
    kTVMFFIRawStr: Result := UTF8String(Arg^.VCStr);
    kTVMFFIStr:
      begin
        ArrayView := PTVMFFIByteArray(PByte(Arg^.VObj) + TVMFFIObjectPayloadOffset);
        RequireCpu(ArrayView^.Size <= High(Integer), 'CUDA source is too large');
        SetString(Result, ArrayView^.Data, ArrayView^.Size);
      end;
    kTVMFFISmallStr: SetString(Result, @Arg^.VBytes[0], Arg^.Padding and $FF);
  else
    raise ECpuGraph.Create('TVM did not supply CUDA source text');
  end;
end;

function CompileCudaCallback(Handle: Pointer; Args: PTVMFFIAny; NumArgs: Int32;
  ResultValue: PTVMFFIAny): Int32; cdecl;
var
  Cubin: TBytes;
  View: TTVMFFIByteArray;
  Error: UTF8String;
begin
  try
    RequireCpu((ActiveCompiler <> nil) and (Args <> nil) and (NumArgs in [1, 2]),
      'No active native CUDA compiler');
    Cubin := ActiveCompiler.CompileSource(SourceString(Args));
    View.Data := PAnsiChar(Pointer(Cubin));
    View.Size := Length(Cubin);
    Result := ActiveCompiler.FBytesFromArray(@View, ResultValue);
  except
    on E: Exception do
    begin
      Error := UTF8Encode(E.Message);
      TTVMRuntime.TModule.TVMFFIErrorSetRaisedFromCStr('RuntimeError', PAnsiChar(Error));
      Result := -1;
    end;
  end;
end;

function CudaHeaderCallback(Handle: Pointer; Args: PTVMFFIAny; NumArgs: Int32;
  ResultValue: PTVMFFIAny): Int32; cdecl;
const
  BaseHeader: UTF8String = '#include <cuda/std/cstdint>'#10 +
    '#include <cuda/std/type_traits>'#10 +
    'using cuda::std::int8_t; using cuda::std::uint8_t;'#10 +
    'using cuda::std::int16_t; using cuda::std::uint16_t;'#10 +
    'using cuda::std::int32_t; using cuda::std::uint32_t;'#10 +
    'using cuda::std::int64_t; using cuda::std::uint64_t;'#10 +
    'using uint = unsigned int;'#10 +
    'namespace std {'#10 +
    'using cuda::std::is_same; using cuda::std::is_same_v;'#10 +
    'using cuda::std::is_integral; using cuda::std::is_signed;'#10 +
    'using cuda::std::is_unsigned; using cuda::std::is_floating_point;'#10 +
    'using cuda::std::enable_if; using cuda::std::conditional;'#10 +
    '}'#10 +
    '#ifndef __asm__'#10'#define __asm__ asm'#10'#endif'#10 +
    '#ifndef __volatile__'#10'#define __volatile__ volatile'#10'#endif'#10;
var
  Header, Tag, MessageText: UTF8String;
  Value: TTVMFFIAny;
  View: TTVMFFIByteArray;
  Error: TTVMCallError;
  I, Count: Int64;
begin
  try
    RequireCpu((ActiveCompiler <> nil) and (Args <> nil) and (NumArgs = 1), 'Invalid CUDA header request');
    RequireTVM(TTVMRuntime.CallGlobal('ffi.ArraySize', [Args^], Value, @Error), @Error);
    RequireCpu(Value.TypeIndex = kTVMFFIInt, 'Invalid CUDA header tag list');
    Count := Value.VInt64;
    Header := BaseHeader;
    for I := 0 to Count - 1 do
    begin
      RequireTVM(TTVMRuntime.CallGlobal('ffi.ArrayGetItem', [Args^, TTVMRuntime.AnyInt(I)], Value, @Error), @Error);
      try
        Tag := SourceString(@Value);
        if Tag = 'math_constants' then
          Header := Header + '#include <math_constants.h>'#10
        else if Tag = 'int8' then
            Header := Header + '#include <sm_61_intrinsics.h>'#10
          else if Tag = 'mma' then
              Header := Header + '#include <mma.h>'#10
            else if Tag = 'cuda/barrier' then
                Header := Header + '#include <cuda/barrier>'#10
              else if Tag = 'cooperative_groups' then
                  Header := Header + '#include <cooperative_groups.h>'#10
                else if (Tag <> 'cuda') and (Tag <> 'warp_shuffle') then
                    raise ECpuGraph.Create('Unsupported CUDA header tag: ' + UTF8Decode(Tag));
      finally
        if Value.TypeIndex >= kTVMFFIStaticObjectBegin then TTVMRuntime.ObjectRelease(Value.VObj);
      end;
    end;
    View.Data := PAnsiChar(Header);
    View.Size := Length(Header);
    Result := ActiveCompiler.FStringFromArray(@View, ResultValue);
  except
    on E: Exception do
    begin
      MessageText := UTF8Encode(E.Message);
      TTVMRuntime.TModule.TVMFFIErrorSetRaisedFromCStr('RuntimeError', PAnsiChar(MessageText));
      Result := -1;
    end;
  end;
end;

function FindCudaIntrinsic(Handle: Pointer; Args: PTVMFFIAny; NumArgs: Int32;
  ResultValue: PTVMFFIAny): Int32; cdecl;
begin
  { Return None so TVM's CUDA generator handles its built-in operators. }
  ResultValue^ := Default(TTVMFFIAny);
  Result := 0;
end;

function TNvrtcCompiler.Entry(const Name: AnsiString): Pointer;
begin
  Result := GetProcAddress(FLibrary, PAnsiChar(Name));
  RequireCpu(Result <> nil, 'Missing NVRTC entry point: ' + string(Name));
end;

constructor TNvrtcCompiler.Create(const LibraryFile, Architecture: string);
var
  FileName, IncludePath, BuiltinsPath: string;
  I, Major, Minor: Integer;
  Version: TGetVersion;
begin
  inherited Create;
  RequireCpu((Length(Architecture) >= 5) and (Copy(Architecture, 1, 3) = 'sm_'),
    'Expected a CUDA architecture such as sm_89');
  for I := 4 to Length(Architecture) do
    RequireCpu(CharInSet(Architecture[I], ['0'..'9']) or ((I = Length(Architecture)) and
      CharInSet(Architecture[I], ['a', 'f'])), 'Invalid CUDA architecture');
  FileName := ExpandFileName(LibraryFile);
  IncludePath := IncludeTrailingPathDelimiter(ExtractFileDir(ExcludeTrailingPathDelimiter(ExtractFileDir(FileName)))) + 'include';
  RequireCpu(DirectoryExists(IncludePath), 'CUDA Toolkit headers not found: ' + IncludePath);
  FArchitecture := UTF8Encode('--gpu-architecture=' + Architecture);
  FIncludePath := UTF8Encode('--include-path=' + IncludePath);
  FLibrary := LoadLibraryExW(PWideChar(FileName), 0,
    LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR or LOAD_LIBRARY_SEARCH_SYSTEM32);
  if FLibrary = 0 then RaiseLastOSError;
  FCreate := TCreate(Entry('nvrtcCreateProgram'));
  FDestroy := TDestroy(Entry('nvrtcDestroyProgram'));
  FCompile := TCompile(Entry('nvrtcCompileProgram'));
  FCubinSize := TGetSize(Entry('nvrtcGetCUBINSize'));
  FCubin := TGetData(Entry('nvrtcGetCUBIN'));
  FLogSize := TGetSize(Entry('nvrtcGetProgramLogSize'));
  FLog := TGetData(Entry('nvrtcGetProgramLog'));
  FError := TGetErrorString(Entry('nvrtcGetErrorString'));
  Version := TGetVersion(Entry('nvrtcVersion'));
  Check(Version(Major, Minor), 'Read NVRTC version');
  BuiltinsPath := ExtractFilePath(FileName) + Format('nvrtc-builtins64_%d%d.dll', [Major, Minor]);
  FBuiltins := LoadLibraryExW(PWideChar(BuiltinsPath), 0,
    LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR or LOAD_LIBRARY_SEARCH_SYSTEM32);
  if FBuiltins = 0 then RaiseLastOSError;
  FBytesFromArray := TBytesFromArray(GetProcAddress(HMODULE(TTVMRuntime.TModule.Handle),
    'TVMFFIBytesFromByteArray'));
  FStringFromArray := TBytesFromArray(GetProcAddress(HMODULE(TTVMRuntime.TModule.Handle),
    'TVMFFIStringFromByteArray'));
  RequireCpu(Assigned(FBytesFromArray) and Assigned(FStringFromArray) and
    Assigned(TTVMRuntime.TModule.TVMFFIErrorSetRaisedFromCStr),
    'The TVM FFI library cannot pass CUDA compiler output');
end;

destructor TNvrtcCompiler.Destroy;
begin
  if FLibrary <> 0 then FreeLibrary(FLibrary);
  if FBuiltins <> 0 then FreeLibrary(FBuiltins);
  inherited;
end;

procedure TNvrtcCompiler.Check(const Status: Integer; const Operation: string);
begin
  if Status <> 0 then
    raise ECpuGraph.Create(Operation + ': ' + UTF8Decode(UTF8String(FError(Status))));
end;

function TNvrtcCompiler.CompileSource(const Source: UTF8String): TBytes;
var
  ProgramHandle: Pointer;
  Options: array[0..3] of PAnsiChar;
  Size: NativeUInt;
  Status: Integer;
  Log: TBytes;
begin
  Result := nil;
  RequireCpu((Length(Source) > 0) and (Pos(#0, Source) = 0), 'Invalid CUDA source text');
  ProgramHandle := nil;
  Check(FCreate(ProgramHandle, PAnsiChar(Source), 'tvm_kernels.cu', 0, nil, nil), 'Create CUDA program');
  try
    Options[0] := PAnsiChar(FArchitecture);
    Options[1] := PAnsiChar(FIncludePath);
    Options[2] := '--std=c++17';
    Options[3] := '--diag-suppress=177';
    Status := FCompile(ProgramHandle, Length(Options), @Options[0]);
    if Status <> 0 then
    begin
      if (FLogSize(ProgramHandle, Size) = 0) and (Size > 0) and (Size <= High(Integer)) then
      begin
        SetLength(Log, Size);
        if FLog(ProgramHandle, PAnsiChar(Pointer(Log))) = 0 then
          raise ECpuGraph.Create(UTF8Decode(UTF8String(PAnsiChar(Pointer(Log)))));
      end;
      Check(Status, 'Compile CUDA program');
    end;
    Check(FCubinSize(ProgramHandle, Size), 'Read compiled CUDA size');
    RequireCpu((Size > 0) and (Size <= High(Integer)), 'Invalid compiled CUDA size');
    SetLength(Result, Size);
    Check(FCubin(ProgramHandle, PAnsiChar(Pointer(Result))), 'Read compiled CUDA program');
  finally
    FDestroy(ProgramHandle);
  end;
end;

function TNvrtcCompiler.BuildDeviceModule(const IR, Target: TTVMFFIObjectHandle): TTVMFFIObjectHandle;
var
  Error: TTVMCallError;
  Value: TTVMFFIAny;
begin
  if not CallbackRegistered then
  begin
    RequireTVM(TTVMRuntime.RegisterGlobalFunc('tvm_callback_cuda_compile', CompileCudaCallback, False, @Error), @Error);
    CallbackRegistered := True;
  end;
  if not HeaderRegistered then
  begin
    RequireTVM(TTVMRuntime.RegisterGlobalFunc('tirx.intrinsics.cuda.header_generator', CudaHeaderCallback, False, @Error), @Error);
    HeaderRegistered := True;
  end;
  if not IntrinsicsRegistered then
  begin
    RequireTVM(TTVMRuntime.RegisterGlobalFunc('tirx.intrinsics.cuda.get_codegen', FindCudaIntrinsic, False, @Error), @Error);
    IntrinsicsRegistered := True;
  end;
  RequireCpu(ActiveCompiler = nil, 'A CUDA compilation is already active on this thread');
  ActiveCompiler := Self;
  try
    RequireTVM(TTVMRuntime.CallGlobal('target.build.cuda', [TTVMRuntime.AnyObject(IR), TTVMRuntime.AnyObject(Target)], Value, @Error), @Error);
    RequireCpu((Value.TypeIndex = kTVMFFIModule) and (Value.VObj <> nil),
      'TVM returned an invalid CUDA module');
    Result := Value.VObj;
  finally
    ActiveCompiler := nil;
  end;
end;

end.
