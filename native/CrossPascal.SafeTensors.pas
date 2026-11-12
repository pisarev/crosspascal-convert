{ ************************************************************************** }
{                                                                            }
{ CrossPascal.SafeTensors                                                    }
{                                                                            }
{ Copyright © 2026 Yuriy Pisarev                                             }
{                                                                            }
{ ************************************************************************** }

unit CrossPascal.SafeTensors;

{$IFDEF FPC}{$MODE DELPHIUNICODE}{$ENDIF}

interface

uses
  {$IFDEF FPC}SysUtils, Classes, Generics.Collections
  {$ELSE}System.SysUtils, System.Classes, System.Generics.Collections{$ENDIF};

type
  ESafeTensors = class(Exception);
  TSafeTensorInfo = record
    DType: string;
    Shape: TArray<Int64>;
    ElementCount: Int64;
  end;

  TSafeTensorFile = class
  private type
    TEntry = record
      Info: TSafeTensorInfo;
      Offset: Int64;
    end;
  private
    FFile: TFileStream;
    FTensors: TDictionary<string, TEntry>;
    function Entry(const Name: string): TEntry;
  public
    constructor Create(const FileName: string);
    destructor Destroy; override;
    function TensorNames: TArray<string>;
    function TensorInfo(const Name: string): TSafeTensorInfo;
    procedure WriteFloat32(const Name: string; const Destination: TStream);
  end;

  TSafeTensorCheckpoint = class
  private
    FFiles: TObjectDictionary<string, TSafeTensorFile>;
    FWeights: TDictionary<string, TSafeTensorFile>;
    function WeightFile(const Name: string): TSafeTensorFile;
    function OpenShard(const Folder, Name: string): TSafeTensorFile;
  public
    constructor Create(const Directory: string);
    destructor Destroy; override;
    function TensorNames: TArray<string>;
    function TensorInfo(const Name: string): TSafeTensorInfo;
    procedure WriteFloat32(const Name: string; const Destination: TStream);
  end;

implementation

uses
  CrossPascal.Json, CrossVision.Float16;

procedure Check(const Condition: Boolean; const MessageText: string);
begin
  if not Condition then raise ESafeTensors.Create(MessageText);
end;

function ByteWidth(const DType: string): Integer;
begin
  if (DType = 'BF16') or (DType = 'F16') or (DType = 'I16') or (DType = 'U16') then Exit(2);
  if (DType = 'F32') or (DType = 'I32') or (DType = 'U32') then Exit(4);
  if (DType = 'F64') or (DType = 'I64') or (DType = 'U64') then Exit(8);
  if (DType = 'U8') or (DType = 'I8') or (DType = 'BOOL') then Exit(1);
  raise ESafeTensors.Create('Unsupported safetensors dtype: ' + DType);
end;

function NonNegative(const Value: TJSONValue; const LabelText: string): Int64;
begin
  Check((Value is TJSONNumber) and TryStrToInt64(JSONValueText(Value), Result) and (Result >= 0),
    'Invalid safetensors ' + LabelText);
end;

constructor TSafeTensorFile.Create(const FileName: string);
const
  MaxHeader = 100000000;
var
  HeaderSize: UInt64;
  Header: UTF8String;
  Parsed, Value: TJSONValue;
  Root, Tensor: TJSONObject;
  Shape, Range: TJSONArray;
  Ranges: TDictionary<Int64, Int64>;
  Info: TEntry;
  Name: string;
  Origin, DataSize, Start, Finish, Dim, Cursor: Int64;
  I, J, Width: Integer;
begin
  inherited Create;
  FTensors := TDictionary<string, TEntry>.Create;
  FFile := TFileStream.Create(FileName, fmOpenRead or fmShareDenyWrite);
  Check(FFile.Size >= 8, 'Safetensors file is missing its header length');
  FFile.ReadBuffer(HeaderSize, SizeOf(HeaderSize));
  Check((HeaderSize > 0) and (HeaderSize <= MaxHeader) and
    (HeaderSize <= UInt64(FFile.Size - 8)), 'Invalid safetensors header length');
  SetLength(Header, Integer(HeaderSize));
  FFile.ReadBuffer(Header[1], Length(Header));
  Check(Header[1] = '{', 'Safetensors header must begin with a JSON object');
  Origin := FFile.Position;
  DataSize := FFile.Size - Origin;
  Parsed := ParseCrossJSON(UTF8Decode(Header));
  try
    Check(Parsed is TJSONObject, 'Invalid safetensors header');
    Root := TJSONObject(Parsed);
    Ranges := TDictionary<Int64, Int64>.Create;
    try
      for I := 0 to Root.Count - 1 do
      begin
        Name := JSONObjectName(Root, I);
        Value := JSONObjectValue(Root, I);
        Check(Value is TJSONObject, 'Invalid safetensors entry: ' + Name);
        if Name = '__metadata__' then Continue;
        Tensor := TJSONObject(Value);
        Value := Tensor.GetValue('dtype');
        Check(Value is TJSONString, 'Missing safetensors dtype: ' + Name);
        Info := Default(TEntry);
        Info.Info.DType := JSONValueText(Value);
        Width := ByteWidth(Info.Info.DType);
        Value := Tensor.GetValue('shape');
        Check(Value is TJSONArray, 'Missing safetensors shape: ' + Name);
        Shape := TJSONArray(Value);
        Check(Shape.Count <= 64, 'Unsupported safetensors rank');
        SetLength(Info.Info.Shape, Shape.Count);
        Info.Info.ElementCount := 1;
        for J := 0 to Shape.Count - 1 do
        begin
          Dim := NonNegative(Shape.Items[J], 'shape dimension');
          Info.Info.Shape[J] := Dim;
          if Dim > 0 then
            Check(Info.Info.ElementCount <= High(Int64) div Width div Dim, 'Safetensors shape is too large');
          Info.Info.ElementCount := Info.Info.ElementCount * Dim;
        end;
        Value := Tensor.GetValue('data_offsets');
        Check(Value is TJSONArray, 'Missing safetensors offsets: ' + Name);
        Range := TJSONArray(Value);
        Check(Range.Count = 2, 'Expected two safetensors offsets');
        Start := NonNegative(Range.Items[0], 'offset');
        Finish := NonNegative(Range.Items[1], 'offset');
        Check((Start <= Finish) and (Finish <= DataSize) and (Finish - Start = Info.Info.ElementCount * Width),
          'Invalid safetensors byte range: ' + Name);
        if Start < Finish then
        begin
          Check(not Ranges.ContainsKey(Start), 'Safetensors byte ranges overlap');
          Ranges.Add(Start, Finish);
        end;
        Info.Offset := Origin + Start;
        Check(not FTensors.ContainsKey(Name), 'Duplicate safetensors tensor: ' + Name);
        FTensors.Add(Name, Info);
      end;
      Cursor := 0;
      while Ranges.TryGetValue(Cursor, Finish) do
      begin
        Ranges.Remove(Cursor);
        Cursor := Finish;
      end;
      Check((Cursor = DataSize) and (Ranges.Count = 0), 'Safetensors byte ranges contain holes or overlap');
    finally
      Ranges.Free;
    end;
  finally
    Parsed.Free;
  end;
end;

destructor TSafeTensorFile.Destroy;
begin
  FTensors.Free;
  FFile.Free;
  inherited;
end;

function TSafeTensorFile.Entry(const Name: string): TEntry;
begin
  if not FTensors.TryGetValue(Name, Result) then
    raise ESafeTensors.Create('Tensor is missing from the checkpoint: ' + Name);
end;

function TSafeTensorFile.TensorNames: TArray<string>;
var
  Name: string;
  I: Integer;
begin
  SetLength(Result, FTensors.Count);
  I := 0;
  for Name in FTensors.Keys do
  begin
    Result[I] := Name;
    Inc(I);
  end;
end;

function TSafeTensorFile.TensorInfo(const Name: string): TSafeTensorInfo;
begin
  Result := Entry(Name).Info;
  Result.Shape := Copy(Result.Shape);
end;

procedure TSafeTensorFile.WriteFloat32(const Name: string; const Destination: TStream);
const
  ChunkElements = 262144;
var
  Tensor: TEntry;
  Input: TArray<Word>;
  Output: TArray<Single>;
  Remaining: Int64;
  Bits: Cardinal;
  Half: TFloat16;
  Count, I: Integer;
begin
  Tensor := Entry(Name);
  Check((Tensor.Info.DType = 'F32') or (Tensor.Info.DType = 'F16') or
    (Tensor.Info.DType = 'BF16'), 'Cannot convert this tensor to float32: ' + Name);
  Remaining := Tensor.Info.ElementCount;
  FFile.Position := Tensor.Offset;
  SetLength(Output, ChunkElements);
  if Tensor.Info.DType <> 'F32' then SetLength(Input, ChunkElements);
  while Remaining > 0 do
  begin
    Count := ChunkElements;
    if Remaining < Count then Count := Integer(Remaining);
    if Tensor.Info.DType = 'F32' then
      FFile.ReadBuffer(Output[0], Count * SizeOf(Single))
    else begin
      FFile.ReadBuffer(Input[0], Count * SizeOf(Word));
      if Tensor.Info.DType = 'BF16' then
        for I := 0 to Count - 1 do
        begin
          Bits := Cardinal(Input[I]) shl 16;
          Move(Bits, Output[I], SizeOf(Single));
        end
      else
        for I := 0 to Count - 1 do
        begin
          Half.Value := Input[I];
          Output[I] := Half.ToSingle;
        end;
    end;
    Destination.WriteBuffer(Output[0], Count * SizeOf(Single));
    Dec(Remaining, Count);
  end;
end;

function TSafeTensorCheckpoint.OpenShard(const Folder, Name: string): TSafeTensorFile;
begin
  Check((Name <> '') and (ExtractFileExt(Name) = '.safetensors') and
    (Pos('/', Name) = 0) and (Pos('\', Name) = 0) and (Pos(':', Name) = 0),
    'Invalid safetensors shard filename: ' + Name);
  if FFiles.TryGetValue(Name, Result) then Exit;
  Result := TSafeTensorFile.Create(Folder + Name);
  try
    FFiles.Add(Name, Result);
  except
    Result.Free;
    raise;
  end;
end;

constructor TSafeTensorCheckpoint.Create(const Directory: string);
var
  Folder, IndexName, Name, ShardName: string;
  Raw: UTF8String;
  FileStream: TFileStream;
  Parsed, Value: TJSONValue;
  Map: TJSONObject;
  Shard, Mapped: TSafeTensorFile;
  I: Integer;
begin
  inherited Create;
  FFiles := TObjectDictionary<string, TSafeTensorFile>.Create([doOwnsValues]);
  FWeights := TDictionary<string, TSafeTensorFile>.Create;
  Folder := IncludeTrailingPathDelimiter(ExpandFileName(Directory));
  IndexName := Folder + 'model.safetensors.index.json';
  if not FileExists(IndexName) then
  begin
    Shard := OpenShard(Folder, 'model.safetensors');
    for Name in Shard.TensorNames do FWeights.Add(Name, Shard);
    Exit;
  end;
  FileStream := TFileStream.Create(IndexName, fmOpenRead or fmShareDenyWrite);
  try
    Check((FileStream.Size > 0) and (FileStream.Size <= 64 * 1024 * 1024),
      'Invalid safetensors index size');
    SetLength(Raw, FileStream.Size);
    FileStream.ReadBuffer(Raw[1], Length(Raw));
  finally
    FileStream.Free;
  end;
  Parsed := ParseCrossJSON(UTF8Decode(Raw));
  try
    Check(Parsed is TJSONObject, 'Invalid safetensors index');
    Value := TJSONObject(Parsed).GetValue('weight_map');
    Check(Value is TJSONObject, 'Missing safetensors weight map');
    Map := TJSONObject(Value);
    Check(Map.Count > 0, 'Empty safetensors weight map');
    for I := 0 to Map.Count - 1 do
    begin
      Name := JSONObjectName(Map, I);
      Value := JSONObjectValue(Map, I);
      Check(Value is TJSONString, 'Invalid shard for tensor: ' + Name);
      ShardName := JSONValueText(Value);
      Shard := OpenShard(Folder, ShardName);
      Shard.Entry(Name);
      Check(not FWeights.ContainsKey(Name), 'Duplicate safetensors weight map entry: ' + Name);
      FWeights.Add(Name, Shard);
    end;
    for Shard in FFiles.Values do
      for Name in Shard.TensorNames do
        Check(FWeights.TryGetValue(Name, Mapped) and (Mapped = Shard),
          'Checkpoint shard and index disagree on tensor: ' + Name);
  finally
    Parsed.Free;
  end;
end;

destructor TSafeTensorCheckpoint.Destroy;
begin
  FWeights.Free;
  FFiles.Free;
  inherited;
end;

function TSafeTensorCheckpoint.WeightFile(const Name: string): TSafeTensorFile;
begin
  if not FWeights.TryGetValue(Name, Result) then
    raise ESafeTensors.Create('Tensor is missing from the checkpoint: ' + Name);
end;

function TSafeTensorCheckpoint.TensorNames: TArray<string>;
var
  Name: string;
  I: Integer;
begin
  SetLength(Result, FWeights.Count);
  I := 0;
  for Name in FWeights.Keys do
  begin
    Result[I] := Name;
    Inc(I);
  end;
end;

function TSafeTensorCheckpoint.TensorInfo(const Name: string): TSafeTensorInfo;
begin
  Result := WeightFile(Name).TensorInfo(Name);
end;

procedure TSafeTensorCheckpoint.WriteFloat32(const Name: string; const Destination: TStream);
begin
  WeightFile(Name).WriteFloat32(Name, Destination);
end;

end.
