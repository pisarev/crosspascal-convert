{ ************************************************************************** }
{                                                                            }
{ CrossPascal.ModelGraph                                                     }
{                                                                            }
{ Copyright © 2026 Yuriy Pisarev (ypisareff@outlook.com)                     }
{                                                                            }
{ ************************************************************************** }

unit CrossPascal.ModelGraph;

{$IFDEF FPC}{$MODE DELPHIUNICODE}{$ENDIF}

interface

uses
  {$IFDEF FPC}SysUtils, Generics.Collections{$ELSE}System.SysUtils, System.Generics.Collections{$ENDIF},
  CrossPascal.Onnx;

type
  TRotaryEmbedding = record
    Cosine: string;
    Sine: string;
  end;

  TModelGraphBuilder = class
  protected
    FInputs: TList<TOnnxValue>;
    FOutputs: TList<TOnnxValue>;
    FNodes: TList<TOnnxNode>;
    FConstants: TList<TOnnxTensor>;
    FData: TList<TBytes>;
    FSerial: Integer;
    function NextName: string;
    function ValueInfo(const Name: string; const Kind: Integer; const Shape: array of Int64): TOnnxValue;
    procedure AddOutput(const Name, Value: string; const Kind: Integer; const Shape: array of Int64);
    function Constant(const Data: Pointer; const Size, Kind: Integer; const Shape: array of Int64): string;
    function Float(const Value: Single): string;
    function IntegerValue(const Value: Int64): string;
    function Integers(const Values: array of Int64): string;
    function Op(const Kind: string; const Inputs: array of string): string;
    function AxisOp(const Kind: string; const Inputs: array of string; const Axis: Integer): string;
    function Reshape(const Input: string; const Shape: array of Int64): string;
    function Transpose(const Input: string; const Order: array of Int64): string;
    function Mean(const Input: string; const Axis: Integer; const Keep: Boolean = True): string;
    function Gather(const Input, Indices: string; const Axis: Integer): string;
    function Part(const Input: string; const Axis, First, AfterLast: Int64): string;
    function Stack(const Items: array of string): string;
    function Cast(const Input: string; const DataType: Integer): string;
    function RMSNormalize(const Input: string; const Epsilon: Single): string;
    function GELU(const Input: string): string;
    function Rotary(const Position: string; const Base: Single;
      const Chunk, HeadSize: Integer): TRotaryEmbedding;
    function ApplyRotary(const Input: string; const Factors: TRotaryEmbedding;
      const HeadSize: Integer): string;
    function RepeatHeads(const Input: string; const Heads, KVHeads, Context, HeadSize: Integer): string;
    function CreateModel: TOnnxModel;
  public
    constructor Create;
    destructor Destroy; override;
  end;

implementation

uses
  {$IFDEF FPC}Math{$ELSE}System.Math{$ENDIF}, TVM.CpuGraph;

constructor TModelGraphBuilder.Create;
begin
  inherited Create;
  FInputs := TList<TOnnxValue>.Create;
  FOutputs := TList<TOnnxValue>.Create;
  FNodes := TList<TOnnxNode>.Create;
  FConstants := TList<TOnnxTensor>.Create;
  FData := TList<TBytes>.Create;
end;

destructor TModelGraphBuilder.Destroy;
begin
  FData.Free;
  FConstants.Free;
  FNodes.Free;
  FOutputs.Free;
  FInputs.Free;
  inherited;
end;

function TModelGraphBuilder.NextName: string;
begin
  Result := 'value_' + IntToStr(FSerial);
  Inc(FSerial);
end;

function TModelGraphBuilder.ValueInfo(const Name: string; const Kind: Integer;
  const Shape: array of Int64): TOnnxValue;
var
  I: Integer;
begin
  Result := Default(TOnnxValue);
  Result.Name := Name;
  Result.DataType := Kind;
  Result.HasShape := True;
  SetLength(Result.Shape, Length(Shape));
  for I := 0 to High(Shape) do Result.Shape[I].Value := Shape[I];
end;

procedure TModelGraphBuilder.AddOutput(const Name, Value: string; const Kind: Integer;
  const Shape: array of Int64);
var
  Node: TOnnxNode;
begin
  Node := Default(TOnnxNode);
  Node.Name := Name;
  Node.OpType := 'Identity';
  Node.Inputs := [Value];
  Node.Outputs := [Name];
  FNodes.Add(Node);
  FOutputs.Add(ValueInfo(Name, Kind, Shape));
end;

function TModelGraphBuilder.Constant(const Data: Pointer; const Size, Kind: Integer;
  const Shape: array of Int64): string;
var
  Bytes: TBytes;
  Tensor: TOnnxTensor;
  I: Integer;
begin
  SetLength(Bytes, Size);
  if Size > 0 then Move(Data^, Bytes[0], Size);
  FData.Add(Bytes);
  Tensor := Default(TOnnxTensor);
  Tensor.Name := NextName;
  Tensor.DataType := Kind;
  SetLength(Tensor.Shape, Length(Shape));
  for I := 0 to High(Shape) do Tensor.Shape[I] := Shape[I];
  Tensor.RawData.Data := Pointer(Bytes);
  Tensor.RawData.Size := Size;
  FConstants.Add(Tensor);
  Result := Tensor.Name;
end;

function TModelGraphBuilder.Float(const Value: Single): string;
begin
  Result := Constant(@Value, SizeOf(Single), 1, []);
end;

function TModelGraphBuilder.IntegerValue(const Value: Int64): string;
begin
  Result := Constant(@Value, SizeOf(Int64), 7, []);
end;

function TModelGraphBuilder.Integers(const Values: array of Int64): string;
begin
  RequireCpu(Length(Values) > 0, 'Expected a nonempty integer constant');
  Result := Constant(@Values[0], Length(Values) * SizeOf(Int64), 7, [Length(Values)]);
end;

function TModelGraphBuilder.Op(const Kind: string; const Inputs: array of string): string;
var
  Node: TOnnxNode;
  I: Integer;
begin
  Node := Default(TOnnxNode);
  Node.Name := NextName;
  Node.OpType := Kind;
  Node.Outputs := [Node.Name];
  SetLength(Node.Inputs, Length(Inputs));
  for I := 0 to High(Inputs) do Node.Inputs[I] := Inputs[I];
  FNodes.Add(Node);
  Result := Node.Name;
end;

function TModelGraphBuilder.AxisOp(const Kind: string; const Inputs: array of string;
  const Axis: Integer): string;
var
  Node: TOnnxNode;
begin
  Result := Op(Kind, Inputs);
  Node := FNodes.Last;
  SetLength(Node.Attributes, 1);
  Node.Attributes[0].Name := 'axis';
  Node.Attributes[0].IntValue := Axis;
  FNodes[FNodes.Count - 1] := Node;
end;

function TModelGraphBuilder.Reshape(const Input: string; const Shape: array of Int64): string;
begin
  Result := Op('Reshape', [Input, Integers(Shape)]);
end;

function TModelGraphBuilder.Transpose(const Input: string; const Order: array of Int64): string;
var
  Node: TOnnxNode;
  I: Integer;
begin
  Result := Op('Transpose', [Input]);
  Node := FNodes.Last;
  SetLength(Node.Attributes, 1);
  Node.Attributes[0].Name := 'perm';
  SetLength(Node.Attributes[0].IntValues, Length(Order));
  for I := 0 to High(Order) do Node.Attributes[0].IntValues[I] := Order[I];
  FNodes[FNodes.Count - 1] := Node;
end;

function TModelGraphBuilder.Mean(const Input: string; const Axis: Integer; const Keep: Boolean): string;
var
  Node: TOnnxNode;
begin
  Result := Op('ReduceMean', [Input, Integers([Axis])]);
  Node := FNodes.Last;
  SetLength(Node.Attributes, 1);
  Node.Attributes[0].Name := 'keepdims';
  Node.Attributes[0].IntValue := Ord(Keep);
  FNodes[FNodes.Count - 1] := Node;
end;

function TModelGraphBuilder.Gather(const Input, Indices: string; const Axis: Integer): string;
begin
  Result := AxisOp('Gather', [Input, Indices], Axis);
end;

function TModelGraphBuilder.Part(const Input: string; const Axis, First, AfterLast: Int64): string;
begin
  Result := Op('Slice', [Input, Integers([First]), Integers([AfterLast]), Integers([Axis])]);
end;

function TModelGraphBuilder.Stack(const Items: array of string): string;
var
  Expanded: TArray<string>;
  Axis: string;
  I: Integer;
begin
  SetLength(Expanded, Length(Items));
  Axis := Integers([0]);
  for I := 0 to High(Items) do Expanded[I] := Op('Unsqueeze', [Items[I], Axis]);
  Result := AxisOp('Concat', Expanded, 0);
end;

function TModelGraphBuilder.Cast(const Input: string; const DataType: Integer): string;
var
  Node: TOnnxNode;
begin
  Result := Op('Cast', [Input]);
  Node := FNodes.Last;
  SetLength(Node.Attributes, 1);
  Node.Attributes[0].Name := 'to';
  Node.Attributes[0].IntValue := DataType;
  FNodes[FNodes.Count - 1] := Node;
end;

function TModelGraphBuilder.RMSNormalize(const Input: string; const Epsilon: Single): string;
var
  Variance: string;
begin
  Variance := Mean(Op('Mul', [Input, Input]), -1);
  Result := Op('Div', [Input, Op('Sqrt', [Op('Add', [Variance, Float(Epsilon)])])]);
end;

function TModelGraphBuilder.GELU(const Input: string): string;
var
  Cube, Inner: string;
begin
  Cube := Op('Mul', [Op('Mul', [Input, Input]), Input]);
  Inner := Op('Mul', [Float(Sqrt(2 / Pi)), Op('Add', [Input, Op('Mul', [Float(0.044715), Cube])])]);
  Result := Op('Mul', [Op('Mul', [Input, Float(0.5)]), Op('Add', [Float(1), Op('Tanh', [Inner])])]);
end;

function TModelGraphBuilder.Rotary(const Position: string; const Base: Single;
  const Chunk, HeadSize: Integer): TRotaryEmbedding;
var
  Inverse: TArray<Single>;
  I, Half: Integer;
  Phase, Both: string;
  Denominator: Single;
begin
  Half := HeadSize div 2;
  RequireCpu((Base > 0) and (Half > 0) and (HeadSize mod 2 = 0), 'Invalid rotary embedding configuration');
  SetLength(Inverse, Half);
  for I := 0 to High(Inverse) do
  begin
    Denominator := Power(Base, I / Half);
    Inverse[I] := 1 / Denominator;
  end;
  Phase := Op('Mul', [Reshape(Cast(Position, 1), [Chunk, 1]),
    Constant(@Inverse[0], Length(Inverse) * SizeOf(Single), 1, [1, Half])]);
  Both := Reshape(AxisOp('Concat', [Phase, Phase], 1), [1, Chunk, 1, HeadSize]);
  Result.Cosine := Op('Cos', [Both]);
  Result.Sine := Op('Sin', [Both]);
end;

function TModelGraphBuilder.ApplyRotary(const Input: string; const Factors: TRotaryEmbedding;
  const HeadSize: Integer): string;
var
  Half: Integer;
  Rotated: string;
begin
  Half := HeadSize div 2;
  Rotated := AxisOp('Concat', [Op('Neg', [Part(Input, 3, Half, HeadSize)]), Part(Input, 3, 0, Half)], 3);
  Result := Op('Add', [Op('Mul', [Input, Factors.Cosine]), Op('Mul', [Rotated, Factors.Sine])]);
end;

function TModelGraphBuilder.RepeatHeads(const Input: string;
  const Heads, KVHeads, Context, HeadSize: Integer): string;
var
  Groups: Integer;
begin
  RequireCpu((KVHeads > 0) and (Heads >= KVHeads) and (Heads mod KVHeads = 0),
    'Invalid grouped attention head counts');
  Groups := Heads div KVHeads;
  if Groups = 1 then Exit(Input);
  Result := Op('Tile', [Op('Unsqueeze', [Input, Integers([2])]), Integers([1, 1, Groups, 1, 1])]);
  Result := Reshape(Result, [1, Heads, Context, HeadSize]);
end;

function TModelGraphBuilder.CreateModel: TOnnxModel;
begin
  Result := TOnnxModel.CreateGenerated(18, FInputs.ToArray, FOutputs.ToArray, FConstants.ToArray,
    FNodes.ToArray, FData.ToArray);
end;

end.
