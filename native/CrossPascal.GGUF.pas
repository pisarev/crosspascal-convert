{ ************************************************************************** }
{                                                                            }
{ CrossPascal.GGUF                                                           }
{                                                                            }
{ Copyright © 2026 Yuriy Pisarev                                             }
{                                                                            }
{ ************************************************************************** }

unit CrossPascal.GGUF;

{$IFDEF FPC}{$MODE DELPHIUNICODE}{$ENDIF}
{$POINTERMATH ON}

interface

uses
  {$IFDEF FPC}Classes, Windows{$ELSE}System.Classes, Winapi.Windows{$ENDIF}, TVM.CpuGraph;

type
  TGGUFTensor = record
    Name: string;
    Dimensions: array[0..3] of Int64;
    Format: Integer;
    DataOffset: Int64;
    DataBytes: Int64;
    Elements: Int64;
  end;

  TGGUFCheckpoint = class
  private type
    TInitParams = record
      NoAlloc: Boolean;
      TensorContext: Pointer;
    end;
    TReader = function(UserData, Output: Pointer; Offset: UInt64; Count: NativeUInt): NativeUInt; cdecl;
    TOpen = function(Reader: TReader; UserData: Pointer; MaxChunk: NativeUInt; FileSize: UInt64;
      Params: TInitParams): Pointer; cdecl;
    TClose = procedure(Context: Pointer); cdecl;
    TContextSize = function(Context: Pointer): NativeUInt; cdecl;
    TFind = function(Context: Pointer; Name: PAnsiChar): Int64; cdecl;
    TCount = function(Context: Pointer): Int64; cdecl;
    TText = function(Context: Pointer; Index: Int64): PAnsiChar; cdecl;
    TArrayText = function(Context: Pointer; Index: Int64; Element: NativeUInt): PAnsiChar; cdecl;
    TShape = function(Context: Pointer; Index: Int64): PInt64; cdecl;
    TTensorSize = function(Context: Pointer; Index: Int64): NativeUInt; cdecl;
    TKind = function(Context: Pointer; Index: Int64): Integer; cdecl;
    TU32 = function(Context: Pointer; Index: Int64): UInt32; cdecl;
    TF32 = function(Context: Pointer; Index: Int64): Single; cdecl;
    TBool = function(Context: Pointer; Index: Int64): Boolean; cdecl;
    TArrayData = function(Context: Pointer; Index: Int64): Pointer; cdecl;
    TBlockSize = function(Kind: Integer): Int64; cdecl;
    TTypeSize = function(Kind: Integer): NativeUInt; cdecl;
    TDecode = procedure(Source: Pointer; Dest: PSingle; Count: Int64); cdecl;
  private
    FLibrary: HMODULE;
    FFile: TMappedModelFile;
    FContext: Pointer;
    FClose: TClose;
    FFindTensor: TFind;
    FFindKey: TFind;
    FTensorCount: TCount;
    FDataOffset: TContextSize;
    FTensorName: TText;
    FValueString: TText;
    FArrayString: TArrayText;
    FTensorShape: TShape;
    FTensorOffset: TTensorSize;
    FTensorSize: TTensorSize;
    FTensorKind: TKind;
    FValueKind: TKind;
    FValueU32: TU32;
    FValueF32: TF32;
    FValueBool: TBool;
    FArrayKind: TKind;
    FArrayCount: TTensorSize;
    FArrayData: TArrayData;
    FBlockSize: TBlockSize;
    FTypeSize: TTypeSize;
    function Entry(const Name: AnsiString): Pointer;
    function Key(const Name: string; const Kind: Integer; const Required: Boolean = True): Int64;
    function ArrayKey(const Name: string; const Kind: Integer; out Count: Integer): Int64;
    function ArrayData(const Name: string; const Kind: Integer; out Count: Integer): Pointer;
    function DecodeFunction(const Kind: Integer): TDecode;
    procedure WriteTensor(const Tensor: TGGUFTensor; const Dest: TStream);
  public
    constructor Create(const FileName, GGMLLibrary: string);
    destructor Destroy; override;
    function Count: Int64;
    function Tensor(const Index: Int64): TGGUFTensor; overload;
    function Tensor(const Name: string): TGGUFTensor; overload;
    function Text(const Name: string): string;
    function Unsigned(const Name: string): UInt32; overload;
    function Unsigned(const Name: string; const DefaultValue: UInt32): UInt32; overload;
    function Float(const Name: string): Single;
    function Flag(const Name: string): Boolean; overload;
    function Flag(const Name: string; const DefaultValue: Boolean): Boolean; overload;
    function TokenPieces(const Name: string): TArray<UTF8String>;
    function IntArray(const Name: string): TArray<Int32>;
    function FloatArray(const Name: string): TArray<Single>;
    function FlagArray(const Name: string): TArray<Boolean>;
    procedure WriteFloat32(const Name: string; const Dest: TStream);
  end;

implementation

uses
  {$IFDEF FPC}SysUtils{$ELSE}System.SysUtils{$ENDIF};

const
  U32Type = 4;
  I32Type = 5;
  F32Type = 6;
  BoolType = 7;
  StringType = 8;
  ArrayType = 9;

function ReadMappedFile(UserData, Output: Pointer; Offset: UInt64; Count: NativeUInt): NativeUInt; cdecl;
var
  Source: TMappedModelFile;
begin
  Result := 0;
  Source := TMappedModelFile(UserData);
  if (Output = nil) or (Offset > UInt64(Source.Size)) then Exit;
  if Count > UInt64(Source.Size) - Offset then Count := UInt64(Source.Size) - Offset;
  if Count = 0 then Exit;
  Move(Source.AtOffset(Int64(Offset), Int64(Count))^, Output^, Count);
  Result := Count;
end;

function TGGUFCheckpoint.Entry(const Name: AnsiString): Pointer;
begin
  Result := GetProcAddress(FLibrary, PAnsiChar(Name));
  RequireCpu(Result <> nil, 'Missing GGML entry point: ' + string(Name));
end;

constructor TGGUFCheckpoint.Create(const FileName, GGMLLibrary: string);
var
  Open: TOpen;
  Params: TInitParams;
begin
  inherited Create;
  FFile := TMappedModelFile.Create(ExpandFileName(FileName));
  FLibrary := LoadLibraryExW(PWideChar(ExpandFileName(GGMLLibrary)), 0,
    LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR or LOAD_LIBRARY_SEARCH_SYSTEM32);
  if FLibrary = 0 then RaiseLastOSError;
  Open := TOpen(Entry('gguf_init_from_callback'));
  FClose := TClose(Entry('gguf_free'));
  FFindTensor := TFind(Entry('gguf_find_tensor'));
  FFindKey := TFind(Entry('gguf_find_key'));
  FTensorCount := TCount(Entry('gguf_get_n_tensors'));
  FDataOffset := TContextSize(Entry('gguf_get_data_offset'));
  FTensorName := TText(Entry('gguf_get_tensor_name'));
  FValueString := TText(Entry('gguf_get_val_str'));
  FTensorShape := TShape(Entry('gguf_get_tensor_ne'));
  FTensorOffset := TTensorSize(Entry('gguf_get_tensor_offset'));
  FTensorSize := TTensorSize(Entry('gguf_get_tensor_size'));
  FTensorKind := TKind(Entry('gguf_get_tensor_type'));
  FValueKind := TKind(Entry('gguf_get_kv_type'));
  FValueU32 := TU32(Entry('gguf_get_val_u32'));
  FValueF32 := TF32(Entry('gguf_get_val_f32'));
  FValueBool := TBool(Entry('gguf_get_val_bool'));
  FArrayString := TArrayText(Entry('gguf_get_arr_str'));
  FArrayKind := TKind(Entry('gguf_get_arr_type'));
  FArrayCount := TTensorSize(Entry('gguf_get_arr_n'));
  FArrayData := TArrayData(Entry('gguf_get_arr_data'));
  FBlockSize := TBlockSize(Entry('ggml_blck_size'));
  FTypeSize := TTypeSize(Entry('ggml_type_size'));
  FillChar(Params, SizeOf(Params), 0);
  Params.NoAlloc := True;
  FContext := Open(@ReadMappedFile, FFile, 16 * 1024 * 1024, FFile.Size, Params);
  RequireCpu(FContext <> nil, 'Could not read GGUF checkpoint: ' + FileName);
  RequireCpu((Count > 0) and (FDataOffset(FContext) <= UInt64(FFile.Size)), 'GGUF has no tensor data');
end;

destructor TGGUFCheckpoint.Destroy;
begin
  if FContext <> nil then FClose(FContext);
  if FLibrary <> 0 then FreeLibrary(FLibrary);
  FFile.Free;
  inherited;
end;

function TGGUFCheckpoint.Count: Int64;
begin
  Result := FTensorCount(FContext);
end;

function TGGUFCheckpoint.Tensor(const Index: Int64): TGGUFTensor;
var
  Shape: PInt64;
  I: Integer;
  DataStart, Offset, Size: UInt64;
begin
  RequireCpu((Index >= 0) and (Index < Count), 'GGUF tensor index is out of range');
  Result.Name := UTF8Decode(UTF8String(FTensorName(FContext, Index)));
  Result.Format := FTensorKind(FContext, Index);
  Shape := FTensorShape(FContext, Index);
  Result.Elements := 1;
  for I := 0 to High(Result.Dimensions) do
  begin
    RequireCpu((Shape[I] > 0) and (Shape[I] <= High(Int64) div Result.Elements), 'Invalid GGUF tensor shape');
    Result.Dimensions[I] := Shape[I];
    Result.Elements := Result.Elements * Shape[I];
  end;
  DataStart := FDataOffset(FContext);
  Offset := FTensorOffset(FContext, Index);
  Size := FTensorSize(FContext, Index);
  RequireCpu((Offset <= UInt64(FFile.Size) - DataStart) and (Size > 0) and
    (Size <= UInt64(FFile.Size) - DataStart - Offset), 'Truncated GGUF tensor: ' + Result.Name);
  Result.DataOffset := DataStart + Offset;
  Result.DataBytes := Size;
end;

function TGGUFCheckpoint.Tensor(const Name: string): TGGUFTensor;
var
  Encoded: UTF8String;
  Index: Int64;
begin
  Encoded := UTF8Encode(Name);
  Index := FFindTensor(FContext, PAnsiChar(Encoded));
  RequireCpu(Index >= 0, 'Missing GGUF tensor: ' + Name);
  Result := Tensor(Index);
end;

function TGGUFCheckpoint.Key(const Name: string; const Kind: Integer; const Required: Boolean): Int64;
var
  Encoded: UTF8String;
begin
  Encoded := UTF8Encode(Name);
  Result := FFindKey(FContext, PAnsiChar(Encoded));
  if Result < 0 then
  begin
    RequireCpu(not Required, 'Missing GGUF metadata: ' + Name);
    Exit;
  end;
  RequireCpu(FValueKind(FContext, Result) = Kind, 'Unexpected GGUF metadata type: ' + Name);
end;

function TGGUFCheckpoint.ArrayKey(const Name: string; const Kind: Integer; out Count: Integer): Int64;
var
  Size: NativeUInt;
begin
  Result := Key(Name, ArrayType);
  RequireCpu(FArrayKind(FContext, Result) = Kind, 'Unexpected GGUF array element type: ' + Name);
  Size := FArrayCount(FContext, Result);
  RequireCpu(Size <= High(Integer) div SizeOf(Single), 'GGUF metadata array is too long: ' + Name);
  Count := Integer(Size);
end;

function TGGUFCheckpoint.ArrayData(const Name: string; const Kind: Integer; out Count: Integer): Pointer;
begin
  Result := FArrayData(FContext, ArrayKey(Name, Kind, Count));
  RequireCpu((Result <> nil) or (Count = 0), 'Missing GGUF array data: ' + Name);
end;

function TGGUFCheckpoint.Text(const Name: string): string;
begin
  Result := UTF8Decode(UTF8String(FValueString(FContext, Key(Name, StringType))));
end;

function TGGUFCheckpoint.Unsigned(const Name: string): UInt32;
begin
  Result := FValueU32(FContext, Key(Name, U32Type));
end;

function TGGUFCheckpoint.Unsigned(const Name: string; const DefaultValue: UInt32): UInt32;
var
  Index: Int64;
begin
  Index := Key(Name, U32Type, False);
  if Index < 0 then Exit(DefaultValue);
  Result := FValueU32(FContext, Index);
end;

function TGGUFCheckpoint.Float(const Name: string): Single;
begin
  Result := FValueF32(FContext, Key(Name, F32Type));
end;

function TGGUFCheckpoint.Flag(const Name: string): Boolean;
begin
  Result := FValueBool(FContext, Key(Name, BoolType));
end;

function TGGUFCheckpoint.Flag(const Name: string; const DefaultValue: Boolean): Boolean;
var
  Index: Int64;
begin
  Index := Key(Name, BoolType, False);
  if Index < 0 then Exit(DefaultValue);
  Result := FValueBool(FContext, Index);
end;

function TGGUFCheckpoint.TokenPieces(const Name: string): TArray<UTF8String>;
var
  Index: Int64;
  Count, I: Integer;
begin
  Index := ArrayKey(Name, StringType, Count);
  Result := nil;
  SetLength(Result, Count);
  for I := 0 to Count - 1 do Result[I] := UTF8String(FArrayString(FContext, Index, I));
end;

function TGGUFCheckpoint.IntArray(const Name: string): TArray<Int32>;
var
  Count: Integer;
  Data: Pointer;
begin
  Data := ArrayData(Name, I32Type, Count);
  Result := nil;
  SetLength(Result, Count);
  if Count > 0 then Move(Data^, Result[0], Count * SizeOf(Int32));
end;

function TGGUFCheckpoint.FloatArray(const Name: string): TArray<Single>;
var
  Count: Integer;
  Data: Pointer;
begin
  Data := ArrayData(Name, F32Type, Count);
  Result := nil;
  SetLength(Result, Count);
  if Count > 0 then Move(Data^, Result[0], Count * SizeOf(Single));
end;

function TGGUFCheckpoint.FlagArray(const Name: string): TArray<Boolean>;
var
  Count, I: Integer;
  Data: PByte;
begin
  Data := ArrayData(Name, BoolType, Count);
  Result := nil;
  SetLength(Result, Count);
  for I := 0 to Count - 1 do Result[I] := Data[I] <> 0;
end;

function TGGUFCheckpoint.DecodeFunction(const Kind: Integer): TDecode;
const
  Names: array[0..30] of AnsiString = ('', 'ggml_fp16_to_fp32_row', 'dequantize_row_q4_0',
    'dequantize_row_q4_1', '', '', 'dequantize_row_q5_0', 'dequantize_row_q5_1',
    'dequantize_row_q8_0', '', 'dequantize_row_q2_K', 'dequantize_row_q3_K', 'dequantize_row_q4_K',
    'dequantize_row_q5_K', 'dequantize_row_q6_K', 'dequantize_row_q8_K', 'dequantize_row_iq2_xxs',
    'dequantize_row_iq2_xs', 'dequantize_row_iq3_xxs', 'dequantize_row_iq1_s', 'dequantize_row_iq4_nl',
    'dequantize_row_iq3_s', 'dequantize_row_iq2_s', 'dequantize_row_iq4_xs', '', '', '', '', '',
    'dequantize_row_iq1_m', 'ggml_bf16_to_fp32_row');
begin
  RequireCpu((Kind > 0) and (Kind <= High(Names)) and (Names[Kind] <> ''),
    Format('Unsupported GGUF tensor encoding %d', [Kind]));
  Result := TDecode(Entry(Names[Kind]));
end;

procedure TGGUFCheckpoint.WriteTensor(const Tensor: TGGUFTensor; const Dest: TStream);
const
  MaxElements = 256 * 1024;
var
  BlockSize, TypeSize, BlocksLeft, BlocksPerChunk, Blocks, Bytes, Elements: Int64;
  Buffer: TArray<Single>;
  Source: PByte;
  Decode: TDecode;
begin
  Source := FFile.AtOffset(Tensor.DataOffset, Tensor.DataBytes);
  BlockSize := FBlockSize(Tensor.Format);
  TypeSize := FTypeSize(Tensor.Format);
  RequireCpu((BlockSize > 0) and (BlockSize <= MaxElements) and (TypeSize > 0) and
    (Tensor.Dimensions[0] mod BlockSize = 0) and (Tensor.DataBytes mod TypeSize = 0),
    'Invalid GGUF block layout: ' + Tensor.Name);
  BlocksLeft := Tensor.Elements div BlockSize;
  RequireCpu(BlocksLeft = Tensor.DataBytes div TypeSize, 'GGUF tensor size mismatch: ' + Tensor.Name);
  BlocksPerChunk := MaxElements div BlockSize;
  Decode := nil;
  if Tensor.Format <> 0 then
  begin
    Decode := DecodeFunction(Tensor.Format);
    SetLength(Buffer, BlocksPerChunk * BlockSize);
  end;
  while BlocksLeft > 0 do
  begin
    Blocks := BlocksLeft;
    if Blocks > BlocksPerChunk then Blocks := BlocksPerChunk;
    Bytes := Blocks * TypeSize;
    Elements := Blocks * BlockSize;
    if Assigned(Decode) then
    begin
      Decode(Source, @Buffer[0], Elements);
      Dest.WriteBuffer(Buffer[0], Elements * SizeOf(Single));
    end
    else
      Dest.WriteBuffer(Source^, Bytes);
    Inc(Source, Bytes);
    Dec(BlocksLeft, Blocks);
  end;
end;

procedure TGGUFCheckpoint.WriteFloat32(const Name: string; const Dest: TStream);
begin
  WriteTensor(Tensor(Name), Dest);
end;

end.
