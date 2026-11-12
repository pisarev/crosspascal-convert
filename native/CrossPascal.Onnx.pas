{ ************************************************************************** }
{                                                                            }
{ CrossPascal.Onnx                                                           }
{                                                                            }
{ Copyright © 2026 Yuriy Pisarev                                             }
{                                                                            }
{ ************************************************************************** }

unit CrossPascal.Onnx;

{$IFDEF FPC}{$MODE DELPHIUNICODE}{$ENDIF}
{$POINTERMATH ON}

interface

uses
  {$IFDEF FPC}SysUtils{$ELSE}System.SysUtils{$ENDIF}, TVM.CpuGraph;

const
  OnnxFloat = 1;
  OnnxInt8 = 3;
  OnnxInt64 = 7;
  OnnxBool = 9;

type
  TOnnxBytes = record
    Data: PByte;
    Size: NativeUInt;
  end;

  TOnnxDimension = record
    Value: Int64;
    Symbol: string;
  end;

  TOnnxValue = record
    Name: string;
    DataType: Integer;
    HasShape: Boolean;
    Shape: TArray<TOnnxDimension>;
  end;

  TOnnxTensor = record
    Name: string;
    DataType: Integer;
    Shape: TArray<Int64>;
    RawData: TOnnxBytes;
    Serialized: TOnnxBytes;
    IsExternal: Boolean;
  end;

  TOnnxAttribute = record
    Name: string;
    Kind: Integer;
    FloatValue: Single;
    IntValue: Int64;
    Text: string;
    Tensor: TOnnxTensor;
    IntValues: TArray<Int64>;
    Serialized: TOnnxBytes;
  end;

  TOnnxNode = record
    Name: string;
    OpType: string;
    Domain: string;
    Inputs: TArray<string>;
    Outputs: TArray<string>;
    Attributes: TArray<TOnnxAttribute>;
  end;

  TOnnxModel = class
  private
    FFile: TMappedModelFile;
    FOpset: Int64;
    FInputs: TArray<TOnnxValue>;
    FOutputs: TArray<TOnnxValue>;
    FValueInfo: TArray<TOnnxValue>;
    FInitializers: TArray<TOnnxTensor>;
    FNodes: TArray<TOnnxNode>;
    FOwnedData: TArray<TBytes>;
    procedure ReadGraph(const Graph: TOnnxBytes);
  public
    constructor Create(const FileName: string);
    constructor CreateGenerated(const Opset: Integer; const Inputs, Outputs: TArray<TOnnxValue>;
      const Initializers: TArray<TOnnxTensor>; const Nodes: TArray<TOnnxNode>;
      const OwnedData: TArray<TBytes>);
    destructor Destroy; override;
    procedure SpecializeInput(const Name: string; const Shape: array of Int64);
    { Raw tensor and attribute views are valid while this model is open. }
    property Opset: Int64 read FOpset;
    property Inputs: TArray < TOnnxValue > read FInputs;
    property Outputs: TArray < TOnnxValue > read FOutputs;
    property ValueInfo: TArray < TOnnxValue > read FValueInfo;
    property Initializers: TArray < TOnnxTensor > read FInitializers;
    property Nodes: TArray < TOnnxNode > read FNodes;
  end;

function ReadOnnxFloat32(const Tensor: TOnnxTensor): TArray<Single>;
function ReadOnnxInt64(const Tensor: TOnnxTensor): TArray<Int64>;
function ReadOnnxBool(const Tensor: TOnnxTensor): TArray<Int64>;

implementation

uses
  {$IFDEF FPC}Generics.Collections{$ELSE}System.Generics.Collections{$ENDIF};

procedure TOnnxModel.SpecializeInput(const Name: string; const Shape: array of Int64);
var
  I, J, Found: Integer;
begin
  TensorElements(Shape);
  Found := -1;
  for I := 0 to High(FInputs) do
    if FInputs[I].Name = Name then
    begin
      RequireCpu(Found < 0, 'Duplicate ONNX input name');
      Found := I;
    end;
  RequireCpu((Found >= 0) and FInputs[Found].HasShape and
    (Length(FInputs[Found].Shape) = Length(Shape)), 'ONNX profile input rank differs');
  for J := 0 to High(Shape) do
    RequireCpu((FInputs[Found].Shape[J].Value < 0) or
      (FInputs[Found].Shape[J].Value = Shape[J]), 'ONNX profile changes a fixed input dimension');
  for J := 0 to High(Shape) do FInputs[Found].Shape[J].Value := Shape[J];
end;

type
  TProtoField = record
    Number: Integer;
    Wire: Integer;
    Bits: UInt64;
    Bytes: TOnnxBytes;
    function NumberValue: Int64;
    function TextValue: string;
    function FloatValue: Single;
    function MessageValue: TOnnxBytes;
  end;

  TProtoReader = record
  private
    FBytes: TOnnxBytes;
    FPosition: NativeUInt;
    function Take(const Size: NativeUInt): TOnnxBytes;
    function ReadVarint: UInt64;
  public
    constructor Create(const Bytes: TOnnxBytes);
    function Next(out Field: TProtoField): Boolean;
    function AtEnd: Boolean;
  end;

constructor TProtoReader.Create(const Bytes: TOnnxBytes);
begin
  FBytes := Bytes;
  FPosition := 0;
end;

function TProtoReader.AtEnd: Boolean;
begin
  Result := FPosition = FBytes.Size;
end;

function TProtoReader.Take(const Size: NativeUInt): TOnnxBytes;
begin
  RequireCpu(Size <= FBytes.Size - FPosition, 'Truncated ONNX message');
  Result.Data := FBytes.Data + FPosition;
  Result.Size := Size;
  Inc(FPosition, Size);
end;

function TProtoReader.ReadVarint: UInt64;
var
  ByteValue: Byte;
  I: Integer;
begin
  Result := 0;
  for I := 0 to 9 do
  begin
    RequireCpu(not AtEnd, 'Truncated ONNX integer');
    ByteValue := FBytes.Data[FPosition];
    Inc(FPosition);
    RequireCpu((I < 9) or (ByteValue <= 1), 'ONNX integer is out of range');
    Result := Result or (UInt64(ByteValue and $7F) shl (7 * I));
    if ByteValue < $80 then Exit;
  end;
end;

function TProtoReader.Next(out Field: TProtoField): Boolean;
var
  Tag, LengthValue: UInt64;
begin
  Result := not AtEnd;
  if not Result then Exit;
  Field := Default(TProtoField);
  Tag := ReadVarint;
  RequireCpu((Tag >= 8) and (Tag <= High(UInt32)), 'Invalid ONNX field number');
  Field.Number := Integer(Tag shr 3);
  Field.Wire := Integer(Tag and 7);
  case Field.Wire of
    0: Field.Bits := ReadVarint;
    1: Field.Bytes := Take(8);
    2:
      begin
        LengthValue := ReadVarint;
        Field.Bytes := Take(LengthValue);
      end;
    5: Field.Bytes := Take(4);
  else
    raise ECpuGraph.Create('Unsupported ONNX wire format');
  end;
end;

function TProtoField.NumberValue: Int64;
begin
  RequireCpu(Wire = 0, 'Expected an ONNX integer');
  Move(Bits, Result, SizeOf(Result));
end;

function TProtoField.MessageValue: TOnnxBytes;
begin
  RequireCpu(Wire = 2, 'Expected an ONNX length-delimited field');
  Result := Bytes;
end;

function TProtoField.TextValue: string;
var
  Payload: TOnnxBytes;
  Encoded: UTF8String;
begin
  Payload := MessageValue;
  RequireCpu(Payload.Size <= High(Integer), 'ONNX string is too long');
  SetString(Encoded, PAnsiChar(Payload.Data), Integer(Payload.Size));
  Result := UTF8Decode(Encoded);
end;

function TProtoField.FloatValue: Single;
begin
  RequireCpu(Wire = 5, 'Expected an ONNX FP32 value');
  Move(Bytes.Data^, Result, SizeOf(Result));
end;

procedure AddIntegers(var Values: TArray<Int64>; const Field: TProtoField);
var
  Reader: TProtoReader;
  PackedBits: UInt64;
  Count: Integer;
begin
  Count := Length(Values);
  if Field.Wire = 0 then
  begin
    SetLength(Values, Count + 1);
    Values[Count] := Field.NumberValue;
    Exit;
  end;
  Reader := TProtoReader.Create(Field.MessageValue);
  while not Reader.AtEnd do
  begin
    if Count = Length(Values) then SetLength(Values, Count + 8);
    PackedBits := Reader.ReadVarint;
    Move(PackedBits, Values[Count], SizeOf(PackedBits));
    Inc(Count);
  end;
  SetLength(Values, Count);
end;

procedure AddText(var Values: TArray<string>; const Field: TProtoField);
var
  Count: Integer;
begin
  Count := Length(Values);
  SetLength(Values, Count + 1);
  Values[Count] := Field.TextValue;
end;

function ReadTensor(const Bytes: TOnnxBytes): TOnnxTensor;
var
  Reader: TProtoReader;
  Field: TProtoField;
begin
  Result := Default(TOnnxTensor);
  Result.Serialized := Bytes;
  Reader := TProtoReader.Create(Bytes);
  while Reader.Next(Field) do
    case Field.Number of
      1: AddIntegers(Result.Shape, Field);
      2: Result.DataType := Field.NumberValue;
      8: Result.Name := Field.TextValue;
      9: Result.RawData := Field.MessageValue;
      14: Result.IsExternal := Field.NumberValue = 1;
    end;
  RequireCpu(Result.DataType > 0, 'ONNX tensor is missing its data type');
end;

function ReadAttribute(const Bytes: TOnnxBytes): TOnnxAttribute;
var
  Reader: TProtoReader;
  Field: TProtoField;
begin
  Result := Default(TOnnxAttribute);
  Result.Serialized := Bytes;
  Reader := TProtoReader.Create(Bytes);
  while Reader.Next(Field) do
    case Field.Number of
      1: Result.Name := Field.TextValue;
      2: Result.FloatValue := Field.FloatValue;
      3: Result.IntValue := Field.NumberValue;
      4: Result.Text := Field.TextValue;
      5: Result.Tensor := ReadTensor(Field.MessageValue);
      8: AddIntegers(Result.IntValues, Field);
      20: Result.Kind := Field.NumberValue;
    end;
  RequireCpu(Result.Name <> '', 'ONNX attribute is missing its name');
end;

function ReadNode(const Bytes: TOnnxBytes): TOnnxNode;
var
  Reader: TProtoReader;
  Field: TProtoField;
  Count: Integer;
begin
  Result := Default(TOnnxNode);
  Reader := TProtoReader.Create(Bytes);
  while Reader.Next(Field) do
    case Field.Number of
      1: AddText(Result.Inputs, Field);
      2: AddText(Result.Outputs, Field);
      3: Result.Name := Field.TextValue;
      4: Result.OpType := Field.TextValue;
      5:
        begin
          Count := Length(Result.Attributes);
          SetLength(Result.Attributes, Count + 1);
          Result.Attributes[Count] := ReadAttribute(Field.MessageValue);
        end;
      7: Result.Domain := Field.TextValue;
    end;
  RequireCpu(Result.OpType <> '', 'ONNX node is missing its operator type');
end;

function ReadShape(const Bytes: TOnnxBytes): TArray<TOnnxDimension>;
var
  Reader, DimReader: TProtoReader;
  Field, DimField: TProtoField;
  Index: Integer;
begin
  Result := nil;
  Reader := TProtoReader.Create(Bytes);
  while Reader.Next(Field) do
    if Field.Number = 1 then
    begin
      Index := Length(Result);
      SetLength(Result, Index + 1);
      Result[Index].Value := -1;
      DimReader := TProtoReader.Create(Field.MessageValue);
      while DimReader.Next(DimField) do
        case DimField.Number of
          1: Result[Index].Value := DimField.NumberValue;
          2: Result[Index].Symbol := DimField.TextValue;
        end;
    end;
end;

procedure ReadValueType(const Bytes: TOnnxBytes; var Value: TOnnxValue);
var
  Reader, TensorReader: TProtoReader;
  Field, TensorField: TProtoField;
begin
  Reader := TProtoReader.Create(Bytes);
  while Reader.Next(Field) do
    if Field.Number = 1 then
    begin
      TensorReader := TProtoReader.Create(Field.MessageValue);
      while TensorReader.Next(TensorField) do
        case TensorField.Number of
          1: Value.DataType := TensorField.NumberValue;
          2:
            begin
              Value.HasShape := True;
              Value.Shape := ReadShape(TensorField.MessageValue);
            end;
        end;
    end;
end;

function ReadValue(const Bytes: TOnnxBytes): TOnnxValue;
var
  Reader: TProtoReader;
  Field: TProtoField;
begin
  Result := Default(TOnnxValue);
  Reader := TProtoReader.Create(Bytes);
  while Reader.Next(Field) do
    case Field.Number of
      1: Result.Name := Field.TextValue;
      2: ReadValueType(Field.MessageValue, Result);
    end;
  RequireCpu(Result.Name <> '', 'ONNX value is missing its name');
end;

function DefaultDomainOpset(const Bytes: TOnnxBytes): Int64;
var
  Reader: TProtoReader;
  Field: TProtoField;
  Domain: string;
begin
  Result := 0;
  Domain := '';
  Reader := TProtoReader.Create(Bytes);
  while Reader.Next(Field) do
    case Field.Number of
      1: Domain := Field.TextValue;
      2: Result := Field.NumberValue;
    end;
  if (Domain <> '') and (Domain <> 'ai.onnx') then Result := 0;
end;

constructor TOnnxModel.Create(const FileName: string);
var
  Source, Graph: TOnnxBytes;
  Reader: TProtoReader;
  Field: TProtoField;
  Version: Int64;
begin
  inherited Create;
  FFile := TMappedModelFile.Create(ExpandFileName(FileName));
  Source.Size := FFile.Size;
  Source.Data := FFile.AtOffset(0, FFile.Size);
  Graph := Default(TOnnxBytes);
  Reader := TProtoReader.Create(Source);
  while Reader.Next(Field) do
    case Field.Number of
      7: Graph := Field.MessageValue;
      8:
        begin
          Version := DefaultDomainOpset(Field.MessageValue);
          if Version > 0 then FOpset := Version;
        end;
    end;
  RequireCpu((Graph.Size > 0) and (FOpset > 0), 'ONNX model is missing a graph or default opset');
  ReadGraph(Graph);
end;

constructor TOnnxModel.CreateGenerated(const Opset: Integer; const Inputs, Outputs: TArray<TOnnxValue>;
  const Initializers: TArray<TOnnxTensor>; const Nodes: TArray<TOnnxNode>; const OwnedData: TArray<TBytes>);
begin
  inherited Create;
  FOpset := Opset;
  FInputs := Inputs;
  FOutputs := Outputs;
  FInitializers := Initializers;
  FNodes := Nodes;
  FOwnedData := OwnedData;
end;

destructor TOnnxModel.Destroy;
begin
  FFile.Free;
  inherited;
end;

procedure TOnnxModel.ReadGraph(const Graph: TOnnxBytes);
var
  Reader: TProtoReader;
  Field: TProtoField;
  Nodes: TList<TOnnxNode>;
  Initializers: TList<TOnnxTensor>;
  Inputs, Outputs, Values: TList<TOnnxValue>;
begin
  Nodes := nil;
  Initializers := nil;
  Inputs := nil;
  Outputs := nil;
  Values := nil;
  try
    Nodes := TList<TOnnxNode>.Create;
    Initializers := TList<TOnnxTensor>.Create;
    Inputs := TList<TOnnxValue>.Create;
    Outputs := TList<TOnnxValue>.Create;
    Values := TList<TOnnxValue>.Create;
    Reader := TProtoReader.Create(Graph);
    while Reader.Next(Field) do
      case Field.Number of
        1: Nodes.Add(ReadNode(Field.MessageValue));
        5: Initializers.Add(ReadTensor(Field.MessageValue));
        11: Inputs.Add(ReadValue(Field.MessageValue));
        12: Outputs.Add(ReadValue(Field.MessageValue));
        13: Values.Add(ReadValue(Field.MessageValue));
        15: raise ECpuGraph.Create('ONNX sparse initializers are not supported');
      end;
    RequireCpu(Outputs.Count > 0, 'ONNX graph has no outputs');
    FNodes := Nodes.ToArray;
    FInitializers := Initializers.ToArray;
    FInputs := Inputs.ToArray;
    FOutputs := Outputs.ToArray;
    FValueInfo := Values.ToArray;
  finally
    Values.Free;
    Outputs.Free;
    Inputs.Free;
    Initializers.Free;
    Nodes.Free;
  end;
end;

function ReadOnnxFloat32(const Tensor: TOnnxTensor): TArray<Single>;
var
  Reader: TProtoReader;
  Field: TProtoField;
  Payload: TOnnxBytes;
  Elements: Int64;
  Count, PackedCount: Integer;
begin
  RequireCpu((Tensor.DataType = 1) and not Tensor.IsExternal,
    'Expected an embedded ONNX FP32 tensor: ' + Tensor.Name);
  Elements := TensorElements(Tensor.Shape);
  RequireCpu(Elements <= High(Integer) div SizeOf(Single), 'ONNX FP32 tensor is too large');
  Result := nil;
  SetLength(Result, Elements);
  if Tensor.RawData.Data <> nil then
  begin
    RequireCpu(Tensor.RawData.Size = UInt64(Elements) * SizeOf(Single), 'Invalid ONNX FP32 tensor length');
    Move(Tensor.RawData.Data^, Result[0], Tensor.RawData.Size);
    Exit;
  end;
  Reader := TProtoReader.Create(Tensor.Serialized);
  Count := 0;
  while Reader.Next(Field) do
    if Field.Number = 4 then
    begin
      if Field.Wire = 5 then
      begin
        RequireCpu(Count < Elements, 'Too many ONNX FP32 tensor values');
        Result[Count] := Field.FloatValue;
        Inc(Count);
      end
      else begin
        Payload := Field.MessageValue;
        RequireCpu((Payload.Size mod SizeOf(Single) = 0) and (Payload.Size <= UInt64(Elements - Count) * SizeOf(Single)),
          'Invalid packed ONNX FP32 tensor length');
        PackedCount := Payload.Size div SizeOf(Single);
        if PackedCount > 0 then Move(Payload.Data^, Result[Count], Payload.Size);
        Inc(Count, PackedCount);
      end;
    end;
  RequireCpu(Count = Elements, 'Missing ONNX FP32 tensor values: ' + Tensor.Name);
end;

function ReadOnnxInt64(const Tensor: TOnnxTensor): TArray<Int64>;
var
  Reader: TProtoReader;
  Field: TProtoField;
  Elements: Int64;
begin
  RequireCpu((Tensor.DataType = 7) and not Tensor.IsExternal,
    'Expected an embedded ONNX int64 tensor: ' + Tensor.Name);
  { A zero-length shape vector is the ONNX representation of a scalar target.
    Keep empty runtime tensors outside this reader's supported contract. }
  if (Length(Tensor.Shape) = 1) and (Tensor.Shape[0] = 0) then
    Elements := 0
  else
    Elements := TensorElements(Tensor.Shape);
  RequireCpu(Elements <= High(Integer) div SizeOf(Int64), 'ONNX int64 tensor is too large');
  Result := nil;
  if Tensor.RawData.Data <> nil then
  begin
    RequireCpu(Tensor.RawData.Size = UInt64(Elements) * SizeOf(Int64), 'Invalid ONNX int64 tensor length');
    SetLength(Result, Elements);
    if Elements > 0 then Move(Tensor.RawData.Data^, Result[0], Tensor.RawData.Size);
    Exit;
  end;
  Reader := TProtoReader.Create(Tensor.Serialized);
  while Reader.Next(Field) do
    if Field.Number = 7 then AddIntegers(Result, Field);
  RequireCpu(Length(Result) = Elements, 'Missing ONNX int64 tensor values: ' + Tensor.Name);
end;

function ReadOnnxBool(const Tensor: TOnnxTensor): TArray<Int64>;
var
  Reader: TProtoReader;
  Field: TProtoField;
  Elements: Int64;
  I: Integer;
begin
  RequireCpu((Tensor.DataType = 9) and not Tensor.IsExternal,
    'Expected an embedded ONNX bool tensor: ' + Tensor.Name);
  Elements := TensorElements(Tensor.Shape);
  RequireCpu(Elements <= High(Integer) div SizeOf(Int64), 'ONNX bool tensor is too large');
  Result := nil;
  if Tensor.RawData.Data <> nil then
  begin
    RequireCpu(Tensor.RawData.Size = UInt64(Elements), 'Invalid ONNX bool tensor length');
    SetLength(Result, Elements);
    for I := 0 to High(Result) do Result[I] := Ord((Tensor.RawData.Data + I)^ <> 0);
    Exit;
  end;
  Reader := TProtoReader.Create(Tensor.Serialized);
  while Reader.Next(Field) do
    if Field.Number = 5 then AddIntegers(Result, Field);
  RequireCpu(Length(Result) = Elements, 'Missing ONNX bool tensor values: ' + Tensor.Name);
  for I := 0 to High(Result) do Result[I] := Ord(Result[I] <> 0);
end;

end.
