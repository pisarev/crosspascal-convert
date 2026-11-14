{ ************************************************************************** }
{                                                                            }
{ CrossPascal.TVMOnnx                                                        }
{                                                                            }
{ Copyright © 2026 Yuriy Pisarev                                             }
{                                                                            }
{ ************************************************************************** }

unit CrossPascal.TVMOnnx;

{$IFDEF FPC}{$MODE DELPHIUNICODE}{$ENDIF}
{$POINTERMATH ON}

interface

uses
  CrossPascal.Onnx, CrossPascal.TVMCompiler;

procedure CompileCpuOnnx(const CompilerDirectory, InputFile, TargetJSON, KernelObject, VMObject: string;
  const BatchSize: Integer = 1); overload;
procedure CompileCpuOnnx(const Model: TOnnxModel;
  const CompilerDirectory, TargetJSON, KernelObject, VMObject: string;
  const BatchSize: Integer = 1); overload;
procedure CompileCudaOnnx(const Model: TOnnxModel;
  const CompilerDirectory, NvrtcLibrary, Architecture, HostTargetJSON, KernelObject, VMObject: string;
  const BatchSize: Integer = 1);
procedure CompileWindowsTvmOnnxLibrary(const Model: TOnnxModel; const Kit: TWindowsTvmKit;
  const LibraryFile: string; const NvrtcLibrary: string = ''; const Architecture: string = '');

implementation

uses
  {$IFDEF FPC}SysUtils, Math, Windows, Generics.Collections{$ELSE}System.SysUtils, System.Math, Winapi.Windows, System.Generics.Collections{$ENDIF},
  DLPack, TVM, TVM.CpuGraph;

type
  TFieldGetter = function(Field: Pointer; Output: PTVMFFIAny): Integer; cdecl;
  PFieldInfo = ^TFieldInfo;
  TFieldInfo = record
    Name: TTVMFFIByteArray;
    Doc: TTVMFFIByteArray;
    Metadata: TTVMFFIByteArray;
    Flags: Int64;
    Size: Int64;
    Alignment: Int64;
    Offset: Int64;
    Getter: TFieldGetter;
    Setter: Pointer;
    DefaultValue: TTVMFFIAny;
    StaticType: Int32;
  end;
  PObjectTypeInfo = ^TObjectTypeInfo;
  TObjectTypeInfo = record
    TypeIndex: Int32;
    Depth: Int32;
    Key: TTVMFFIByteArray;
    Ancestors: Pointer;
    KeyHash: UInt64;
    FieldCount: Int32;
    MethodCount: Int32;
    Fields: PFieldInfo;
    Methods: Pointer;
    Metadata: Pointer;
  end;
  TGetTypeInfo = function(TypeIndex: Int32): PObjectTypeInfo; cdecl;

  TGraphValue = record
    Expression: TTVMFFIAny;
    Shape: TArray<Int64>;
    DataType: Integer;
    StaticInt: Boolean;
    StaticFloat: Boolean;
    Transposed: Boolean;
    Integers: TArray<Int64>;
    Floats: TArray<Single>;
  end;

  TOnnxBuilder = class
  private
    FObjects: TList<TTVMFFIObjectHandle>;
    FValues: TDictionary<string, TGraphValue>;
    FInitializers: TDictionary<string, TOnnxTensor>;
    FDimensions: TDictionary<string, Int64>;
    FMatMulInputs: TDictionary<string, Boolean>;
    FBatchSize: Integer;
    FCuda: Boolean;
    FOpset: Int64;
    FEmpty: TTVMFFIAny;
    FFloatType: TTVMFFIAny;
    FIntType: TTVMFFIAny;
    FZero: TTVMFFIAny;
    FSumReducer: TTVMFFIAny;
    FBuilder: TTVMFFIAny;
    FAttributes: TTVMFFIAny;
    FFloatDType: TDLDataType;
    FIntDType: TDLDataType;
    FKernelCount: Integer;
    function Make(const Name: string; const Args: array of TTVMFFIAny): TTVMFFIAny;
    procedure Invoke(const Name: string; const Args: array of TTVMFFIAny);
    function IntImm(const Value: Int64): TTVMFFIAny;
    function ShapeArray(const Shape: array of Int64): TTVMFFIAny;
    function IndexArray(const Values: array of Int64): TTVMFFIAny;
    function DataType(const Kind: Integer): TDLDataType;
    function PrimitiveType(const Kind: Integer): TTVMFFIAny;
    function TensorType(const Shape: array of Int64; const Kind: Integer = OnnxFloat): TTVMFFIAny;
    function Placeholder(const Shape: array of Int64; const Name: UTF8String;
      const Kind: Integer = OnnxFloat): TTVMFFIAny;
    function Iteration(const Name: UTF8String; const Extent: Int64; const Reduction: Boolean;
      out Variable: TTVMFFIAny): TTVMFFIAny;
    function Primitive(const Name: string; const Left, Right: TTVMFFIAny): TTVMFFIAny;
    function ReadTensor(const Tensor: TTVMFFIAny; const Indices: array of TTVMFFIAny): TTVMFFIAny;
    function ReduceSum(const Value: TTVMFFIAny; const Axes: array of TTVMFFIAny): TTVMFFIAny;
    function ComputeTensor(const Name: PAnsiChar; const Value: TTVMFFIAny;
      const Axes: array of TTVMFFIAny): TTVMFFIAny;
    function Intrinsic(const Name: PAnsiChar; const Kind: TTVMFFIAny;
      const Args: array of TTVMFFIAny): TTVMFFIAny;
    function ConstantValue(const Values: array of Single; const Shape: TArray<Int64>): TGraphValue;
    function StaticIntegers(const Shape, Values: TArray<Int64>; const Kind: Integer = OnnxInt64): TGraphValue;
    function StaticInput(const Node: TOnnxNode; const Index: Integer): TArray<Int64>;
    function ExpressionOf(const Value: TGraphValue): TTVMFFIAny;
    function Input(const Node: TOnnxNode; const Index: Integer): TGraphValue;
    function ParallelKernel(const Kernel: TTVMFFIAny; const BlockName: PAnsiChar;
      const AxisCount: Integer): TTVMFFIAny;
    function VectorReductionKernel(const Kernel: TTVMFFIAny; const BlockName: PAnsiChar;
      const SpatialRank, VectorWidth: Integer): TTVMFFIAny;
    function ReductionCudaKernel(const Kernel: TTVMFFIAny; const BlockName: PAnsiChar;
      const AxisCount: Integer): TTVMFFIAny;
    function CudaKernel(const Kernel: TTVMFFIAny): TTVMFFIAny;
    function GetField(const Value: TTVMFFIAny; const Name: UTF8String): TTVMFFIAny;
    function ArrayLength(const Value: TTVMFFIAny): Integer;
    function MakeSchedule(const Kernel: TTVMFFIAny): TTVMFFIAny;
    function ScheduledFunction(const Schedule: TTVMFFIAny): TTVMFFIAny;
    function EmitKernel(const Tensors, Inputs, OutputTypes: array of TTVMFFIAny;
      const ParallelBlock: PAnsiChar = nil; const ParallelAxes: Integer = 0;
      const ReduceOnCuda: Boolean = False; const CpuVectorWidth: Integer = 0;
      const CpuSpatialRank: Integer = 0): TTVMFFIAny;
    function Emit(const InputValues: array of TGraphValue; const TensorInputs: array of TTVMFFIAny;
      const Compute: TTVMFFIAny; const Shape: TArray<Int64>; const Kind: Integer = OnnxFloat;
      const ParallelBlock: PAnsiChar = nil; const ParallelAxes: Integer = 0;
      const ReduceOnCuda: Boolean = False): TGraphValue;
    function Conv(const Node: TOnnxNode): TGraphValue;
    function ConvTranspose(const Node: TOnnxNode): TGraphValue;
    function PatchConv3D(const Node: TOnnxNode; const Image, Weights: TGraphValue): TGraphValue;
    function Pool(const Node: TOnnxNode): TGraphValue;
    function Concat(const Node: TOnnxNode): TGraphValue;
    function BatchNormalization(const Node: TOnnxNode): TGraphValue;
    function Binary(const Node: TOnnxNode; const OperatorName: string): TGraphValue;
    function Gemm(const Node: TOnnxNode): TGraphValue;
    function MatMul(const Node: TOnnxNode): TGraphValue;
    function Q8MatMul(const Node: TOnnxNode): TGraphValue;
    function Transpose(const Node: TOnnxNode): TGraphValue;
    function Softmax(const Node: TOnnxNode): TGraphValue;
    function Resize(const Node: TOnnxNode): TGraphValue;
    function ResizeLinear(const Node: TOnnxNode): TGraphValue;
    function Slice(const Node: TOnnxNode): TGraphValue;
    procedure Split(const Node: TOnnxNode);
    procedure AddConstant(const Node: TOnnxNode);
    function EmitReshape(const Source: TGraphValue; const Shape: TArray<Int64>): TGraphValue;
    function Reshape(const Node: TOnnxNode): TGraphValue;
    function Flatten(const Node: TOnnxNode): TGraphValue;
    function Unary(const Node: TOnnxNode; const OperatorName: string): TGraphValue;
    function ShapeOf(const Node: TOnnxNode): TGraphValue;
    function Range(const Node: TOnnxNode): TGraphValue;
    function Unsqueeze(const Node: TOnnxNode): TGraphValue;
    function Squeeze(const Node: TOnnxNode): TGraphValue;
    function Gather(const Node: TOnnxNode; const Elementwise: Boolean): TGraphValue;
    function Reduce(const Node: TOnnxNode): TGraphValue;
    function Pad(const Node: TOnnxNode): TGraphValue;
    function Cast(const Node: TOnnxNode): TGraphValue;
    function ExpandOrTile(const Node: TOnnxNode): TGraphValue;
    function ConstantOfShape(const Node: TOnnxNode): TGraphValue;
    function LayerNormalization(const Node: TOnnxNode): TGraphValue;
    function Clip(const Node: TOnnxNode): TGraphValue;
    function Where(const Node: TOnnxNode): TGraphValue;
    function GridSample(const Node: TOnnxNode): TGraphValue;
    function ScatterND(const Node: TOnnxNode): TGraphValue;
    procedure CudaTopK(const Node: TOnnxNode; const Source: TGraphValue; const Axis: Integer;
      const Count: Int64);
    procedure TopK(const Node: TOnnxNode);
    procedure AddNode(const Node: TOnnxNode);
  public
    constructor Create(const BatchSize: Integer; const Cuda: Boolean = False);
    destructor Destroy; override;
    function Build(const Model: TOnnxModel): TTVMFFIObjectHandle;
  end;

function IntAttribute(const Node: TOnnxNode; const Name: string; const DefaultValue: Int64): Int64;
var
  Attribute: TOnnxAttribute;
begin
  for Attribute in Node.Attributes do
    if Attribute.Name = Name then Exit(Attribute.IntValue);
  Result := DefaultValue;
end;

function FloatAttribute(const Node: TOnnxNode; const Name: string; const DefaultValue: Single): Single;
var
  Attribute: TOnnxAttribute;
begin
  for Attribute in Node.Attributes do
    if Attribute.Name = Name then Exit(Attribute.FloatValue);
  Result := DefaultValue;
end;

function StringAttribute(const Node: TOnnxNode; const Name, DefaultValue: string): string;
var
  Attribute: TOnnxAttribute;
begin
  for Attribute in Node.Attributes do
    if Attribute.Name = Name then Exit(Attribute.Text);
  Result := DefaultValue;
end;

function IntsAttribute(const Node: TOnnxNode; const Name: string;
  const Defaults: array of Int64): TArray<Int64>;
var
  Attribute: TOnnxAttribute;
  I: Integer;
begin
  for Attribute in Node.Attributes do
    if Attribute.Name = Name then
    begin
      RequireCpu((Length(Defaults) = 0) or (Length(Attribute.IntValues) = Length(Defaults)),
        'Unexpected ONNX attribute length: ' + Name);
      Exit(Attribute.IntValues);
    end;
  Result := nil;
  SetLength(Result, Length(Defaults));
  for I := 0 to High(Defaults) do Result[I] := Defaults[I];
end;

procedure RequireExplicitPadding(const Node: TOnnxNode);
var
  Attribute: TOnnxAttribute;
begin
  for Attribute in Node.Attributes do
    if Attribute.Name = 'auto_pad' then
      RequireCpu(Attribute.Text = 'NOTSET', 'Unsupported ONNX automatic padding: ' + Node.Name);
end;

function SpatialPads(const Node: TOnnxNode;
  const InputShape, Kernel, Strides, Dilation: array of Int64): TArray<Int64>;
var
  Mode: string;
  I: Integer;
  OutputSize, Total: Int64;
  Attribute: TOnnxAttribute;
begin
  Mode := StringAttribute(Node, 'auto_pad', 'NOTSET');
  if Mode = 'NOTSET' then Exit(IntsAttribute(Node, 'pads', [0, 0, 0, 0]));
  RequireCpu((Mode = 'VALID') or (Mode = 'SAME_UPPER') or (Mode = 'SAME_LOWER'),
    'Unsupported ONNX automatic padding: ' + Mode);
  for Attribute in Node.Attributes do
    RequireCpu(Attribute.Name <> 'pads', 'ONNX automatic and explicit padding cannot be combined');
  RequireCpu((Length(InputShape) = 4) and (Length(Kernel) = 2) and
    (Length(Strides) = 2) and (Length(Dilation) = 2), 'Invalid automatic padding dimensions');
  SetLength(Result, 4);
  if Mode = 'VALID' then Exit;
  for I := 0 to 1 do
  begin
    RequireCpu((InputShape[I + 2] > 0) and (InputShape[I + 2] <= High(Integer)) and
      (Strides[I] > 0) and (Strides[I] <= 4096) and
      (Dilation[I] > 0) and (Dilation[I] <= 4096) and
      (Kernel[I] > 0) and (Kernel[I] <= 4096), 'Invalid automatic padding extent');
    OutputSize := (InputShape[I + 2] + Strides[I] - 1) div Strides[I];
    Total := (OutputSize - 1) * Strides[I] + Dilation[I] * (Kernel[I] - 1) + 1 - InputShape[I + 2];
    if Total < 0 then Total := 0;
    Result[I] := Total div 2;
    if Mode = 'SAME_LOWER' then Result[I] := Total - Result[I];
    Result[I + 2] := Total - Result[I];
  end;
end;

function SpatialSize(const InputSize, BeforePad, AfterPad, Kernel, Stride, Dilation: Int64): Int64;
var
  Numerator: Int64;
begin
  RequireCpu(
    (InputSize > 0) and (InputSize <= High(Integer)) and (BeforePad >= 0) and (BeforePad <= High(Integer)) and (AfterPad >= 0) and (AfterPad <= High(Integer)) and (Kernel > 0) and (Kernel <= High(Integer)) and (Dilation > 0) and (Dilation <= High(Integer)) and (Stride > 0) and (Stride <= High(Integer)),
    'Invalid ONNX spatial dimensions'
  );
  Numerator := InputSize + BeforePad + AfterPad - Dilation * (Kernel - 1) - 1;
  RequireCpu(Numerator >= 0, 'ONNX kernel is larger than the padded input');
  Result := Numerator div Stride + 1;
end;

constructor TOnnxBuilder.Create(const BatchSize: Integer; const Cuda: Boolean);
var
  EmptyMap, Module: TTVMFFIAny;
begin
  inherited Create;
  RequireCpu(BatchSize > 0, 'ONNX batch size must be positive');
  FBatchSize := BatchSize;
  FCuda := Cuda;
  FObjects := TList<TTVMFFIObjectHandle>.Create;
  FValues := TDictionary<string, TGraphValue>.Create;
  FInitializers := TDictionary<string, TOnnxTensor>.Create;
  FDimensions := TDictionary<string, Int64>.Create;
  FMatMulInputs := TDictionary<string, Boolean>.Create;
  FFloatDType.code := Ord(kDLFloat);
  FFloatDType.bits := 32;
  FFloatDType.lanes := 1;
  FIntDType.code := Ord(kDLInt);
  FIntDType.bits := 64;
  FIntDType.lanes := 1;
  FFloatType := Make('ir.PrimType', [TTVMRuntime.AnyDataType(FFloatDType)]);
  FIntType := Make('ir.PrimType', [TTVMRuntime.AnyDataType(FIntDType)]);
  FZero := Make('ir.FloatImm', [TTVMRuntime.AnyDataType(FFloatDType), TTVMRuntime.AnyFloat(0), FEmpty]);
  EmptyMap := Make('ffi.Map', []);
  Module := Make('ir.IRModule', [EmptyMap, EmptyMap, EmptyMap]);
  FAttributes := Make('ir.Module_GetAttrs', [Module]);
  FBuilder := Make('relax.BlockBuilderCreate', [Module]);
end;

destructor TOnnxBuilder.Destroy;
var
  I: Integer;
  Handle: TTVMFFIObjectHandle;
begin
  if FObjects <> nil then
    for I := FObjects.Count - 1 downto 0 do
    begin
      Handle := FObjects[I];
      TTVMRuntime.ObjectRelease(Handle);
    end;
  FObjects.Free;
  FMatMulInputs.Free;
  FDimensions.Free;
  FInitializers.Free;
  FValues.Free;
  inherited;
end;

function TOnnxBuilder.Make(const Name: string; const Args: array of TTVMFFIAny): TTVMFFIAny;
var
  Handle: TTVMFFIObjectHandle;
begin
  Handle := CompilerObject(Name, Args);
  try
    FObjects.Add(Handle);
    Result := TTVMRuntime.AnyObject(Handle);
  except
    TTVMRuntime.ObjectRelease(Handle);
    raise;
  end;
end;

procedure TOnnxBuilder.Invoke(const Name: string; const Args: array of TTVMFFIAny);
var
  Reply: TTVMFFIAny;
  Error: TTVMCallError;
begin
  if not TTVMRuntime.CallGlobal(Name, Args, Reply, @Error) then
    raise ECpuGraph.Create(Name + ': ' + string(Error));
end;

function TOnnxBuilder.IntImm(const Value: Int64): TTVMFFIAny;
begin
  Result := Make('ir.IntImm', [TTVMRuntime.AnyDataType(FIntDType), TTVMRuntime.AnyInt(Value), FEmpty]);
end;

function TOnnxBuilder.ShapeArray(const Shape: array of Int64): TTVMFFIAny;
var
  Dims: TArray<TTVMFFIAny>;
  I: Integer;
begin
  SetLength(Dims, Length(Shape));
  for I := 0 to High(Shape) do Dims[I] := IntImm(Shape[I]);
  Result := Make('ffi.Array', Dims);
end;

function TOnnxBuilder.IndexArray(const Values: array of Int64): TTVMFFIAny;
var
  Items: TArray<TTVMFFIAny>;
  I: Integer;
begin
  SetLength(Items, Length(Values));
  for I := 0 to High(Values) do Items[I] := TTVMRuntime.AnyInt(Values[I]);
  Result := Make('ffi.Array', Items);
end;

function TOnnxBuilder.DataType(const Kind: Integer): TDLDataType;
begin
  case Kind of
    OnnxFloat: Result := FFloatDType;
    OnnxInt8:
      begin
        Result := FIntDType;
        Result.bits := 8;
      end;
    OnnxInt64: Result := FIntDType;
    OnnxBool:
      begin
        Result.code := Ord(kDLBool);
        Result.bits := 8;
        Result.lanes := 1;
      end;
  else
    raise ECpuGraph.Create('Unsupported ONNX data type: ' + IntToStr(Kind));
  end;
end;

function TOnnxBuilder.PrimitiveType(const Kind: Integer): TTVMFFIAny;
begin
  case Kind of
    OnnxFloat: Result := FFloatType;
    OnnxInt64: Result := FIntType;
  else
    Result := Make('ir.PrimType', [TTVMRuntime.AnyDataType(DataType(Kind))]);
  end;
end;

function TOnnxBuilder.TensorType(const Shape: array of Int64; const Kind: Integer): TTVMFFIAny;
var
  Expression: TTVMFFIAny;
begin
  Expression := Make('relax.ShapeExpr', [ShapeArray(Shape), FEmpty]);
  Result := Make('relax.TensorType',
    [Expression, PrimitiveType(Kind), TTVMRuntime.AnyInt(-1), FEmpty, FEmpty]);
end;

function TOnnxBuilder.Placeholder(const Shape: array of Int64; const Name: UTF8String;
  const Kind: Integer): TTVMFFIAny;
begin
  Result := Make('te.Placeholder',
    [ShapeArray(Shape), PrimitiveType(Kind), TTVMRuntime.AnyCStr(PAnsiChar(Name))]);
end;

function TOnnxBuilder.Iteration(const Name: UTF8String; const Extent: Int64; const Reduction: Boolean;
  out Variable: TTVMFFIAny): TTVMFFIAny;
var
  Domain: TTVMFFIAny;
  Kind: Integer;
begin
  Kind := 0;
  if Reduction then Kind := 2;
  Variable := Make('ir.Var', [TTVMRuntime.AnyCStr(PAnsiChar(Name)), FIntType, FEmpty]);
  Domain := Make('ir.Range_from_min_extent', [IntImm(0), IntImm(Extent), FEmpty]);
  Result := Make('tirx.IterVar',
    [Domain, Variable, TTVMRuntime.AnyInt(Kind), TTVMRuntime.AnyCStr(''), FEmpty]);
end;

function TOnnxBuilder.Primitive(const Name: string; const Left, Right: TTVMFFIAny): TTVMFFIAny;
begin
  Result := Make(Name, [Left, Right, FEmpty]);
end;

function TOnnxBuilder.ReadTensor(const Tensor: TTVMFFIAny; const Indices: array of TTVMFFIAny): TTVMFFIAny;
begin
  Result := Make('tirx.ProducerLoad', [Tensor, Make('ffi.Array', Indices), FEmpty]);
end;

function TOnnxBuilder.ReduceSum(const Value: TTVMFFIAny; const Axes: array of TTVMFFIAny): TTVMFFIAny;
var
  Left, Right: TTVMFFIAny;
begin
  if FSumReducer.TypeIndex = kTVMFFINone then
  begin
    Left := Make('ir.Var', [TTVMRuntime.AnyCStr('left'), FFloatType, FEmpty]);
    Right := Make('ir.Var', [TTVMRuntime.AnyCStr('right'), FFloatType, FEmpty]);
    FSumReducer := Make('tirx.CommReducer', [Make('ffi.Array', [Left]), Make('ffi.Array', [Right]),
      Make('ffi.Array', [Primitive('tirx.Add', Left, Right)]), Make('ffi.Array', [FZero]), FEmpty]);
  end;
  Result := Make('tirx.Reduce', [FSumReducer, Make('ffi.Array', [Value]), Make('ffi.Array', Axes),
    FEmpty, TTVMRuntime.AnyInt(0), Make('ffi.Array', []), FEmpty]);
end;

function TOnnxBuilder.ComputeTensor(const Name: PAnsiChar; const Value: TTVMFFIAny;
  const Axes: array of TTVMFFIAny): TTVMFFIAny;
var
  Operation: TTVMFFIAny;
begin
  Operation := Make('te.ComputeOp', [TTVMRuntime.AnyCStr(Name), TTVMRuntime.AnyCStr(''), FEmpty,
    Make('ffi.Array', Axes), Make('ffi.Array', [Value])]);
  Result := Make('te.OpGetOutput', [Operation, TTVMRuntime.AnyInt(0)]);
end;

function TOnnxBuilder.Intrinsic(const Name: PAnsiChar; const Kind: TTVMFFIAny;
  const Args: array of TTVMFFIAny): TTVMFFIAny;
var
  Op: TTVMFFIAny;
begin
  Op := Make('ir.GetOp', [TTVMRuntime.AnyCStr(Name)]);
  Result := Make('ir.Call', [Kind, Op, Make('ffi.Array', Args), FEmpty, Make('ffi.Array', []), FEmpty]);
end;

function TOnnxBuilder.ConstantValue(const Values: array of Single; const Shape: TArray<Int64>): TGraphValue;
const
  StaticLimit = 65536;
var
  Data: TTVMFFIObjectHandle;
  I: Integer;
begin
  Result := Default(TGraphValue);
  RequireCpu(Length(Values) = TensorElements(Shape), 'Wrong ONNX constant length');
  Data := CpuTensor(Shape, kDLFloat, 32, @Values[0]);
  try
    Result.Expression := Make('relax.Constant', [TTVMRuntime.AnyObject(Data), FEmpty, FEmpty]);
    Result.Shape := Shape;
    Result.DataType := OnnxFloat;
    if Length(Values) <= StaticLimit then
    begin
      Result.StaticFloat := True;
      SetLength(Result.Floats, Length(Values));
      for I := 0 to High(Values) do Result.Floats[I] := Values[I];
    end;
  finally
    TTVMRuntime.ObjectRelease(Data);
  end;
end;

function TOnnxBuilder.StaticIntegers(const Shape, Values: TArray<Int64>; const Kind: Integer): TGraphValue;
begin
  Result := Default(TGraphValue);
  Result.Shape := Shape;
  Result.Integers := Values;
  Result.DataType := Kind;
  Result.StaticInt := True;
end;

function TOnnxBuilder.StaticInput(const Node: TOnnxNode; const Index: Integer): TArray<Int64>;
var
  Value: TGraphValue;
begin
  Value := Input(Node, Index);
  RequireCpu(Value.StaticInt and (Value.DataType = OnnxInt64), 'ONNX input needs constant integers');
  Result := Copy(Value.Integers);
end;

function TOnnxBuilder.ExpressionOf(const Value: TGraphValue): TTVMFFIAny;
var
  Data: TTVMFFIObjectHandle;
  Bytes: TBytes;
  I: Integer;
begin
  if not Value.StaticInt then Exit(Value.Expression);
  RequireCpu(Length(Value.Integers) = TensorElements(Value.Shape), 'Wrong ONNX integer constant length');
  if Value.DataType = OnnxBool then
  begin
    SetLength(Bytes, Length(Value.Integers));
    for I := 0 to High(Bytes) do Bytes[I] := Ord(Value.Integers[I] <> 0);
    Data := CpuTensor(Value.Shape, kDLBool, 8, Pointer(Bytes));
  end
  else
    Data := CpuTensor(Value.Shape, kDLInt, 64, Pointer(Value.Integers));
  try
    Result := Make('relax.Constant', [TTVMRuntime.AnyObject(Data), FEmpty, FEmpty]);
  finally
    TTVMRuntime.ObjectRelease(Data);
  end;
end;

function TOnnxBuilder.Input(const Node: TOnnxNode; const Index: Integer): TGraphValue;
var
  Name: string;
  Initializer: TOnnxTensor;
begin
  RequireCpu((Index >= 0) and (Index < Length(Node.Inputs)), 'Missing ONNX input: ' + Node.Name);
  Name := Node.Inputs[Index];
  if FValues.TryGetValue(Name, Result) then Exit;
  RequireCpu(FInitializers.TryGetValue(Name, Initializer), 'Unknown ONNX input: ' + Name);
  case Initializer.DataType of
    OnnxFloat: Result := ConstantValue(ReadOnnxFloat32(Initializer), Initializer.Shape);
    OnnxInt64: Result := StaticIntegers(Initializer.Shape, ReadOnnxInt64(Initializer));
    OnnxBool: Result := StaticIntegers(Initializer.Shape, ReadOnnxBool(Initializer), OnnxBool);
  else
    raise ECpuGraph.Create('Unsupported ONNX initializer type for ' + Name + ': ' + IntToStr(Initializer.DataType));
  end;
  FValues.Add(Name, Result);
end;

function TOnnxBuilder.MakeSchedule(const Kernel: TTVMFFIAny): TTVMFFIAny;
var
  EmptyMap, Module, Global: TTVMFFIAny;
begin
  EmptyMap := Make('ffi.Map', []);
  Module := Make('ir.IRModule', [EmptyMap, EmptyMap, EmptyMap]);
  Global := Make('ir.GlobalVar', [TTVMRuntime.AnyCStr('main')]);
  Module := Make('ir.Module_Add', [Module, Global, Kernel, TTVMRuntime.AnyBool(False)]);
  Result := Make('s_tir.schedule.TracedSchedule',
    [Module, TTVMRuntime.AnyInt(1), TTVMRuntime.AnyInt(0), TTVMRuntime.AnyInt(1), TTVMRuntime.AnyBool(True)]);
end;

function TOnnxBuilder.ScheduledFunction(const Schedule: TTVMFFIAny): TTVMFFIAny;
begin
  Result := Make('ir.Module_Lookup_str',
    [Make('s_tir.schedule.ScheduleGetMod', [Schedule]), TTVMRuntime.AnyCStr('main')]);
end;

function TOnnxBuilder.GetField(const Value: TTVMFFIAny; const Name: UTF8String): TTVMFFIAny;
var
  GetTypeInfo: TGetTypeInfo;
  Info: PObjectTypeInfo;
  Field: PFieldInfo;
  I: Integer;
begin
  GetTypeInfo := TGetTypeInfo(GetProcAddress(HMODULE(TTVMRuntime.TModule.Handle), 'TVMFFIGetTypeInfo'));
  RequireCpu(Assigned(GetTypeInfo) and (Value.VObj <> nil), 'TVM compiler reflection is unavailable');
  Info := GetTypeInfo(Value.TypeIndex);
  RequireCpu(Info <> nil, 'TVM compiler object type is unknown');
  for I := 0 to Info^.FieldCount - 1 do
  begin
    Field := Info^.Fields + I;
    if (Field^.Name.Size = NativeUInt(Length(Name))) and
      CompareMem(Field^.Name.Data, PAnsiChar(Name), Length(Name)) then
      begin
        Result := Default(TTVMFFIAny);
        RequireCpu(Assigned(Field^.Getter) and (Field^.Getter(PByte(Value.VObj) + Field^.Offset, @Result) = 0),
          'Could not read TVM field: ' + UTF8Decode(Name));
        if Result.TypeIndex >= kTVMFFIStaticObjectBegin then FObjects.Add(Result.VObj);
        Exit;
      end;
  end;
  raise ECpuGraph.Create('TVM compiler object has no field: ' + UTF8Decode(Name));
end;

function TOnnxBuilder.ArrayLength(const Value: TTVMFFIAny): Integer;
var
  Count: TTVMFFIAny;
  Error: TTVMCallError;
begin
  RequireTVM(TTVMRuntime.CallGlobal('ffi.ArraySize', [Value], Count, @Error), @Error);
  RequireCpu((Count.TypeIndex = kTVMFFIInt) and (Count.VInt64 >= 0) and (Count.VInt64 <= High(Integer)),
    'Invalid TVM compiler array');
  Result := Count.VInt64;
end;

function TOnnxBuilder.CudaKernel(const Kernel: TTVMFFIAny): TTVMFFIAny;
var
  Schedule, Root, Blocks, Block: TTVMFFIAny;
  I: Integer;

  procedure BindBlock(const Candidate: TTVMFFIAny);
  var
    ChildBlocks, Loops, Statement, Iterators, Iter, Kind, SpatialLoop, Split, BlockIndex, ThreadIndex: TTVMFFIAny;
    Spatial: TArray<TTVMFFIAny>;
    J, Count, IteratorCount, Used: Integer;
    AllSpatial: Boolean;
  begin
    ChildBlocks := Make('s_tir.schedule.ScheduleGetChildBlocks', [Schedule, Candidate]);
    if ArrayLength(ChildBlocks) <> 0 then
    begin
      for J := 0 to ArrayLength(ChildBlocks) - 1 do
        BindBlock(Make('ffi.ArrayGetItem', [ChildBlocks, TTVMRuntime.AnyInt(J)]));
      Exit;
    end;
    Loops := Make('s_tir.schedule.ScheduleGetLoops', [Schedule, Candidate]);
    Statement := Make('s_tir.schedule.ScheduleGet', [Schedule, Candidate]);
    Iterators := GetField(Statement, 'iter_vars');
    Count := ArrayLength(Loops);
    IteratorCount := ArrayLength(Iterators);
    AllSpatial := True;
    for J := 0 to IteratorCount - 1 do
    begin
      Iter := Make('ffi.ArrayGetItem', [Iterators, TTVMRuntime.AnyInt(J)]);
      Kind := GetField(Iter, 'iter_type');
      RequireCpu(Kind.TypeIndex = kTVMFFIInt, 'Invalid TVM loop kind');
      if Kind.VInt64 <> 0 then AllSpatial := False;
    end;
    RequireCpu(AllSpatial or (Count = IteratorCount), 'CUDA reduction requires a loop per block iterator');
    SetLength(Spatial, Count);
    Used := 0;
    for J := 0 to Count - 1 do
    begin
      if not AllSpatial then
      begin
        Iter := Make('ffi.ArrayGetItem', [Iterators, TTVMRuntime.AnyInt(J)]);
        if GetField(Iter, 'iter_type').VInt64 <> 0 then Continue;
      end;
      Spatial[Used] := Make('ffi.ArrayGetItem', [Loops, TTVMRuntime.AnyInt(J)]);
      Inc(Used);
    end;
    SetLength(Spatial, Used);
    case Used of
      0:
        begin
          SpatialLoop := Candidate;
          if Count > 0 then SpatialLoop := Make('ffi.ArrayGetItem', [Loops, TTVMRuntime.AnyInt(0)]);
          SpatialLoop := Make('s_tir.schedule.ScheduleAddUnitLoop', [Schedule, SpatialLoop]);
        end;
      1: SpatialLoop := Spatial[0];
    else
      SpatialLoop := Make('s_tir.schedule.ScheduleFuse',
        [Schedule, Make('ffi.Array', Spatial), TTVMRuntime.AnyBool(False)]);
    end;
    Split := Make('s_tir.schedule.ScheduleSplit',
      [Schedule, SpatialLoop, Make('ffi.Array', [FEmpty, IntImm(128)]), TTVMRuntime.AnyBool(False), TTVMRuntime.AnyBool(False)]);
    BlockIndex := Make('ffi.ArrayGetItem', [Split, TTVMRuntime.AnyInt(0)]);
    ThreadIndex := Make('ffi.ArrayGetItem', [Split, TTVMRuntime.AnyInt(1)]);
    Invoke('s_tir.schedule.ScheduleBind', [Schedule, BlockIndex, TTVMRuntime.AnyCStr('blockIdx.x')]);
    Invoke('s_tir.schedule.ScheduleBind', [Schedule, ThreadIndex, TTVMRuntime.AnyCStr('threadIdx.x')]);
  end;

begin
  Schedule := MakeSchedule(Kernel);
  Root := Make('s_tir.schedule.ScheduleGetSBlock',
    [Schedule, TTVMRuntime.AnyCStr('root'), TTVMRuntime.AnyCStr('main')]);
  Blocks := Make('s_tir.schedule.ScheduleGetChildBlocks', [Schedule, Root]);
  RequireCpu(ArrayLength(Blocks) > 0, 'CUDA graph has no computation blocks');
  for I := 0 to ArrayLength(Blocks) - 1 do
  begin
    Block := Make('ffi.ArrayGetItem', [Blocks, TTVMRuntime.AnyInt(I)]);
    BindBlock(Block);
  end;
  Result := Make('ir.BaseFuncWithAttr', [ScheduledFunction(Schedule), TTVMRuntime.AnyCStr('tirx.is_scheduled'), TTVMRuntime.AnyBool(True)]);
end;

function TOnnxBuilder.ParallelKernel(const Kernel: TTVMFFIAny; const BlockName: PAnsiChar;
  const AxisCount: Integer): TTVMFFIAny;
var
  Schedule, Block, Loops, Fused: TTVMFFIAny;
  OuterLoops: TArray<TTVMFFIAny>;
  I: Integer;
begin
  Schedule := MakeSchedule(Kernel);
  Block := Make('s_tir.schedule.ScheduleGetSBlock',
    [Schedule, TTVMRuntime.AnyCStr(BlockName), TTVMRuntime.AnyCStr('main')]);
  Loops := Make('s_tir.schedule.ScheduleGetLoops', [Schedule, Block]);
  SetLength(OuterLoops, AxisCount);
  for I := 0 to High(OuterLoops) do
    OuterLoops[I] := Make('ffi.ArrayGetItem', [Loops, TTVMRuntime.AnyInt(I)]);
  Fused := OuterLoops[0];
  if AxisCount > 1 then
    Fused := Make('s_tir.schedule.ScheduleFuse',
      [Schedule, Make('ffi.Array', OuterLoops), TTVMRuntime.AnyBool(True)]);
  Invoke('s_tir.schedule.ScheduleParallel', [Schedule, Fused]);
  Result := ScheduledFunction(Schedule);
end;

function TOnnxBuilder.VectorReductionKernel(const Kernel: TTVMFFIAny; const BlockName: PAnsiChar;
  const SpatialRank, VectorWidth: Integer): TTVMFFIAny;
var
  Schedule, Block, Loops, Split, Outer, Inner, Rows: TTVMFFIAny;
  Spatial, Reduction, Order: TArray<TTVMFFIAny>;
  I, Count: Integer;
begin
  Schedule := MakeSchedule(Kernel);
  Block := Make('s_tir.schedule.ScheduleGetSBlock',
    [Schedule, TTVMRuntime.AnyCStr(BlockName), TTVMRuntime.AnyCStr('main')]);
  Loops := Make('s_tir.schedule.ScheduleGetLoops', [Schedule, Block]);
  Count := ArrayLength(Loops);
  RequireCpu((SpatialRank >= 2) and (Count > SpatialRank), 'Invalid CPU vector reduction rank');
  SetLength(Spatial, SpatialRank - 1);
  for I := 0 to High(Spatial) do
    Spatial[I] := Make('ffi.ArrayGetItem', [Loops, TTVMRuntime.AnyInt(I)]);
  SetLength(Reduction, Count - SpatialRank);
  for I := 0 to High(Reduction) do
    Reduction[I] := Make('ffi.ArrayGetItem', [Loops, TTVMRuntime.AnyInt(SpatialRank + I)]);
  Inner := Make('ffi.ArrayGetItem', [Loops, TTVMRuntime.AnyInt(SpatialRank - 1)]);
  Rows := Spatial[0];
  if Length(Spatial) > 1 then
    Rows := Make('s_tir.schedule.ScheduleFuse',
      [Schedule, Make('ffi.Array', Spatial), TTVMRuntime.AnyBool(True)]);
  Split := Make('s_tir.schedule.ScheduleSplit',
    [Schedule, Inner, Make('ffi.Array', [FEmpty, IntImm(VectorWidth)]), TTVMRuntime.AnyBool(True), TTVMRuntime.AnyBool(False)]);
  Outer := Make('ffi.ArrayGetItem', [Split, TTVMRuntime.AnyInt(0)]);
  Inner := Make('ffi.ArrayGetItem', [Split, TTVMRuntime.AnyInt(1)]);
  SetLength(Order, Length(Reduction) + 3);
  Order[0] := Rows;
  Order[1] := Outer;
  for I := 0 to High(Reduction) do Order[I + 2] := Reduction[I];
  Order[High(Order)] := Inner;
  Invoke('s_tir.schedule.ScheduleReorder', [Schedule, Make('ffi.Array', Order)]);
  Invoke('s_tir.schedule.ScheduleVectorize', [Schedule, Inner]);
  Invoke('s_tir.schedule.ScheduleParallel', [Schedule, Rows]);
  Make('s_tir.schedule.ScheduleDecomposeReduction', [Schedule, Block, Reduction[0]]);
  Result := ScheduledFunction(Schedule);
end;

function TOnnxBuilder.ReductionCudaKernel(const Kernel: TTVMFFIAny; const BlockName: PAnsiChar;
  const AxisCount: Integer): TTVMFFIAny;
var
  Schedule, Block, Loops, Row, Reduction, Split, Lane: TTVMFFIAny;
  Spatial: TArray<TTVMFFIAny>;
  I: Integer;
begin
  Schedule := MakeSchedule(Kernel);
  Block := Make('s_tir.schedule.ScheduleGetSBlock',
    [Schedule, TTVMRuntime.AnyCStr(BlockName), TTVMRuntime.AnyCStr('main')]);
  Loops := Make('s_tir.schedule.ScheduleGetLoops', [Schedule, Block]);
  SetLength(Spatial, AxisCount);
  for I := 0 to High(Spatial) do
    Spatial[I] := Make('ffi.ArrayGetItem', [Loops, TTVMRuntime.AnyInt(I)]);
  Reduction := Make('ffi.ArrayGetItem', [Loops, TTVMRuntime.AnyInt(AxisCount)]);
  Row := Make('s_tir.schedule.ScheduleFuse',
    [Schedule, Make('ffi.Array', Spatial), TTVMRuntime.AnyBool(True)]);
  Split := Make('s_tir.schedule.ScheduleSplit',
    [Schedule, Reduction, Make('ffi.Array', [FEmpty, IntImm(128)]), TTVMRuntime.AnyBool(True), TTVMRuntime.AnyBool(False)]);
  Lane := Make('ffi.ArrayGetItem', [Split, TTVMRuntime.AnyInt(1)]);
  Invoke('s_tir.schedule.ScheduleBind', [Schedule, Row, TTVMRuntime.AnyCStr('blockIdx.x')]);
  Invoke('s_tir.schedule.ScheduleBind', [Schedule, Lane, TTVMRuntime.AnyCStr('threadIdx.x')]);
  Result := Make('ir.BaseFuncWithAttr', [ScheduledFunction(Schedule), TTVMRuntime.AnyCStr('tirx.is_scheduled'), TTVMRuntime.AnyBool(True)]);
end;

function TOnnxBuilder.EmitKernel(const Tensors, Inputs, OutputTypes: array of TTVMFFIAny;
  const ParallelBlock: PAnsiChar; const ParallelAxes: Integer; const ReduceOnCuda: Boolean;
  const CpuVectorWidth, CpuSpatialRank: Integer): TTVMFFIAny;
var
  Kernel, GlobalVar, Args, Call: TTVMFFIAny;
  Name: UTF8String;
begin
  Kernel := Make('te.CreatePrimFunc', [Make('ffi.Array', Tensors), FEmpty]);
  if FCuda then
  begin
    if ReduceOnCuda then
      Kernel := ReductionCudaKernel(Kernel, ParallelBlock, ParallelAxes)
    else
      Kernel := CudaKernel(Kernel);
  end
  else if CpuVectorWidth > 0 then
      Kernel := VectorReductionKernel(Kernel, ParallelBlock, CpuSpatialRank, CpuVectorWidth)
    else if ParallelAxes > 0 then
        Kernel := ParallelKernel(Kernel, ParallelBlock, ParallelAxes);
  Name := UTF8Encode('kernel_' + IntToStr(FKernelCount));
  Inc(FKernelCount);
  Kernel := Make('ir.BaseFuncWithAttr',
    [Kernel, TTVMRuntime.AnyCStr('global_symbol'), TTVMRuntime.AnyCStr(PAnsiChar(Name))]);
  GlobalVar := Make('relax.BlockBuilderAddFunction',
    [FBuilder, Kernel, TTVMRuntime.AnyCStr(PAnsiChar(Name))]);
  Args := Make('relax.Tuple', [Make('ffi.Array', Inputs), FEmpty]);
  Call := Make('relax.op.call_tir', [GlobalVar, Args, Make('ffi.Array', OutputTypes)]);
  Result := Make('relax.BlockBuilderEmit', [FBuilder, Call, TTVMRuntime.AnyCStr('value')]);
end;

function TOnnxBuilder.Emit(const InputValues: array of TGraphValue; const TensorInputs: array of TTVMFFIAny;
  const Compute: TTVMFFIAny; const Shape: TArray<Int64>; const Kind: Integer; const ParallelBlock: PAnsiChar;
  const ParallelAxes: Integer; const ReduceOnCuda: Boolean): TGraphValue;
var
  PrimArgs, RelaxArgs: TArray<TTVMFFIAny>;
  I, VectorWidth: Integer;
begin
  Result := Default(TGraphValue);
  RequireCpu(Length(InputValues) = Length(TensorInputs), 'Mismatched TVM graph arguments');
  SetLength(PrimArgs, Length(TensorInputs) + 1);
  SetLength(RelaxArgs, Length(InputValues));
  for I := 0 to High(TensorInputs) do
  begin
    PrimArgs[I] := TensorInputs[I];
    RelaxArgs[I] := ExpressionOf(InputValues[I]);
  end;
  PrimArgs[High(PrimArgs)] := Compute;
  VectorWidth := 0;
  if not FCuda and (ParallelAxes > 0) and (ParallelBlock <> nil) and
    ((StrComp(ParallelBlock, 'matmul') = 0) or (StrComp(ParallelBlock, 'conv2d') = 0)) then
    begin
      VectorWidth := 8;
      while (VectorWidth > 1) and (Shape[High(Shape)] mod VectorWidth <> 0) do
        VectorWidth := VectorWidth div 2;
      if VectorWidth = 1 then VectorWidth := 0;
    end;
  Result.Expression := EmitKernel(PrimArgs, RelaxArgs, [TensorType(Shape, Kind)], ParallelBlock,
    ParallelAxes, ReduceOnCuda, VectorWidth, Length(Shape));
  Result.Shape := Shape;
  Result.DataType := Kind;
end;

function TOnnxBuilder.Conv(const Node: TOnnxNode): TGraphValue;
const
  AxisNames: array[0..6] of UTF8String = ('n', 'oc', 'h', 'w', 'ic', 'kh', 'kw');
var
  Image, Weights, Bias, Convolved: TGraphValue;
  InputTensor, WeightTensor, BiasTensor, Padded, InputChannel, GroupIndex,
    ImageValue, WeightValue, Product, Y, X, Compute: TTVMFFIAny;
  Shape, Pads, Strides, Dilation, PaddedShape: TArray<Int64>;
  Iter, Axis: array[0..6] of TTVMFFIAny;
  Sizes: array[0..6] of Int64;
  Groups, WindowElements: Int64;
  I, ParallelAxes: Integer;

  function Coordinate(const Output, Kernel: TTVMFFIAny; const Stride, Step: Int64): TTVMFFIAny;
  begin
    Result := Primitive('tirx.Add', Primitive('tirx.Mul', Output, IntImm(Stride)),
      Primitive('tirx.Mul', Kernel, IntImm(Step)));
  end;

begin
  Image := Input(Node, 0);
  Weights := Input(Node, 1);
  if Length(Image.Shape) = 5 then
  begin
    RequireExplicitPadding(Node);
    Exit(PatchConv3D(Node, Image, Weights));
  end;
  Groups := IntAttribute(Node, 'group', 1);
  RequireCpu((Length(Image.Shape) = 4) and (Length(Weights.Shape) = 4) and (Groups > 0) and
    (Image.Shape[1] mod Groups = 0) and (Weights.Shape[0] mod Groups = 0) and
    (Image.Shape[1] div Groups = Weights.Shape[1]), 'Invalid ONNX Conv channel groups');
  Strides := IntsAttribute(Node, 'strides', [1, 1]);
  Dilation := IntsAttribute(Node, 'dilations', [1, 1]);
  Pads := SpatialPads(Node, Image.Shape, [Weights.Shape[2], Weights.Shape[3]], Strides, Dilation);
  SetLength(Shape, 4);
  Shape[0] := Image.Shape[0];
  Shape[1] := Weights.Shape[0];
  Shape[2] := SpatialSize(Image.Shape[2], Pads[0], Pads[2], Weights.Shape[2], Strides[0], Dilation[0]);
  Shape[3] := SpatialSize(Image.Shape[3], Pads[1], Pads[3], Weights.Shape[3], Strides[1], Dilation[1]);
  InputTensor := Placeholder(Image.Shape, 'image');
  WeightTensor := Placeholder(Weights.Shape, 'filters');
  Padded := InputTensor;
  if Pads[0] + Pads[1] + Pads[2] + Pads[3] <> 0 then
  begin
    Padded := Make('topi.nn.pad', [InputTensor, ShapeArray([0, 0, Pads[0], Pads[1]]),
      ShapeArray([0, 0, Pads[2], Pads[3]]), FZero]);
    if FCuda then
    begin
      PaddedShape := Copy(Image.Shape);
      Inc(PaddedShape[2], Pads[0] + Pads[2]);
      Inc(PaddedShape[3], Pads[1] + Pads[3]);
      Image := Emit([Image], [InputTensor], Padded, PaddedShape);
      InputTensor := Placeholder(PaddedShape, 'image');
      Padded := InputTensor;
    end;
  end;
  for I := 0 to 3 do Sizes[I] := Shape[I];
  for I := 4 to 6 do Sizes[I] := Weights.Shape[I - 3];
  for I := 0 to 6 do Axis[I] := Iteration(AxisNames[I], Sizes[I], I >= 4, Iter[I]);
  Y := Coordinate(Iter[2], Iter[5], Strides[0], Dilation[0]);
  X := Coordinate(Iter[3], Iter[6], Strides[1], Dilation[1]);
  InputChannel := Iter[4];
  if Groups > 1 then
  begin
    GroupIndex := Primitive('tirx.FloorDiv', Iter[1], IntImm(Shape[1] div Groups));
    InputChannel := Primitive('tirx.Add',
      Primitive('tirx.Mul', GroupIndex, IntImm(Weights.Shape[1])), Iter[4]);
  end;
  ImageValue := ReadTensor(Padded, [Iter[0], InputChannel, Y, X]);
  WeightValue := ReadTensor(WeightTensor, [Iter[1], Iter[4], Iter[5], Iter[6]]);
  Product := Primitive('tirx.Mul', ImageValue, WeightValue);
  Compute := ComputeTensor('conv2d', ReduceSum(Product, [Axis[4], Axis[5], Axis[6]]),
    [Axis[0], Axis[1], Axis[2], Axis[3]]);
  ParallelAxes := 0;
  WindowElements := TensorElements([Sizes[4], Sizes[5], Sizes[6]]);
  if TensorElements(Shape) >= (2 * 1024 * 1024 + WindowElements - 1) div WindowElements then
    ParallelAxes := 2;
  if (Length(Node.Inputs) > 2) and (Node.Inputs[2] <> '') then
  begin
    Bias := Input(Node, 2);
    RequireCpu((Length(Bias.Shape) = 1) and (Bias.Shape[0] = Shape[1]), 'Invalid ONNX Conv bias');
    BiasTensor := Placeholder(Bias.Shape, 'bias');
    if FCuda then
    begin
      Convolved := Emit([Image, Weights], [InputTensor, WeightTensor], Compute, Shape);
      InputTensor := Placeholder(Shape, 'convolution');
      Compute := Make('topi.nn.bias_add', [InputTensor, BiasTensor, TTVMRuntime.AnyInt(1)]);
      Exit(Emit([Convolved, Bias], [InputTensor, BiasTensor], Compute, Shape));
    end;
    Compute := Make('topi.nn.bias_add', [Compute, BiasTensor, TTVMRuntime.AnyInt(1)]);
    Result := Emit([Image, Weights, Bias], [InputTensor, WeightTensor, BiasTensor], Compute, Shape,
      OnnxFloat, 'conv2d', ParallelAxes);
  end
  else
    Result := Emit([Image, Weights], [InputTensor, WeightTensor], Compute, Shape, OnnxFloat,
      'conv2d', ParallelAxes);
end;

function TOnnxBuilder.ConvTranspose(const Node: TOnnxNode): TGraphValue;
const
  Names: array[0..6] of UTF8String = ('n', 'oc', 'h', 'w', 'ic', 'kh', 'kw');
var
  Source, Weights, Bias, Convolved: TGraphValue;
  Shape, Pads, Strides, Dilation, Extra, Kernel: TArray<Int64>;
  Iter, Axes: array[0..6] of TTVMFFIAny;
  Sizes: array[0..6] of Int64;
  X, W, B, Channel, GroupIndex, Y, Z, InsideY, InsideZ, Pixel, Product, Compute: TTVMFFIAny;
  Groups, Channels: Int64;
  I: Integer;

  function Coordinate(const Output, K: TTVMFFIAny; const Pad, Step, Stride, Extent: Int64;
    out Inside: TTVMFFIAny): TTVMFFIAny;
  var
    Shift: TTVMFFIAny;
  begin
    Shift := Primitive('tirx.Sub', Primitive('tirx.Add', Output, IntImm(Pad)),
      Primitive('tirx.Mul', K, IntImm(Step)));
    Result := Primitive('tirx.FloorDiv', Shift, IntImm(Stride));
    Inside := Primitive('tirx.And',
      Primitive('tirx.EQ', Primitive('tirx.FloorMod', Shift, IntImm(Stride)), IntImm(0)),
      Primitive('tirx.And', Primitive('tirx.LE', IntImm(0), Result),
        Primitive('tirx.LT', Result, IntImm(Extent))));
  end;

begin
  RequireExplicitPadding(Node);
  RequireCpu(Length(IntsAttribute(Node, 'output_shape', [])) = 0,
    'Explicit ConvTranspose output_shape is not supported');
  Source := Input(Node, 0);
  Weights := Input(Node, 1);
  Groups := IntAttribute(Node, 'group', 1);
  RequireCpu((Length(Source.Shape) = 4) and (Length(Weights.Shape) = 4) and
    (Source.DataType = OnnxFloat) and (Weights.DataType = OnnxFloat) and (Groups > 0) and
    (Source.Shape[1] = Weights.Shape[0]) and (Source.Shape[1] mod Groups = 0),
    'Invalid ONNX ConvTranspose channels');
  Pads := IntsAttribute(Node, 'pads', [0, 0, 0, 0]);
  Strides := IntsAttribute(Node, 'strides', [1, 1]);
  Dilation := IntsAttribute(Node, 'dilations', [1, 1]);
  Extra := IntsAttribute(Node, 'output_padding', [0, 0]);
  Kernel := IntsAttribute(Node, 'kernel_shape', [Weights.Shape[2], Weights.Shape[3]]);
  RequireCpu((Length(Pads) = 4) and (Length(Strides) = 2) and
    (Length(Dilation) = 2) and (Length(Extra) = 2) and (Length(Kernel) = 2),
    'Invalid ONNX ConvTranspose attribute rank');
  Shape := [Source.Shape[0], Weights.Shape[1] * Groups, 0, 0];
  Channels := Source.Shape[1] div Groups;
  for I := 0 to 1 do
  begin
    RequireCpu((Strides[I] > 0) and (Strides[I] <= 4096) and (Dilation[I] > 0) and
      (Dilation[I] <= 4096) and (Pads[I] >= 0) and (Pads[I + 2] >= 0) and
      (Extra[I] >= 0) and (Extra[I] < Strides[I]) and
      (Kernel[I] = Weights.Shape[I + 2]), 'Invalid ONNX ConvTranspose spatial attributes');
    Shape[I + 2] := Strides[I] * (Source.Shape[I + 2] - 1) + Extra[I] +
      Dilation[I] * (Kernel[I] - 1) + 1 - Pads[I] - Pads[I + 2];
  end;
  TensorElements(Shape);
  for I := 0 to 3 do Sizes[I] := Shape[I];
  Sizes[4] := Channels;
  Sizes[5] := Kernel[0];
  Sizes[6] := Kernel[1];
  for I := 0 to 6 do Axes[I] := Iteration(Names[I], Sizes[I], I >= 4, Iter[I]);
  X := Placeholder(Source.Shape, 'image');
  W := Placeholder(Weights.Shape, 'weights');
  GroupIndex := Primitive('tirx.FloorDiv', Iter[1], IntImm(Weights.Shape[1]));
  Channel := Primitive('tirx.Add', Primitive('tirx.Mul', GroupIndex, IntImm(Channels)), Iter[4]);
  Y := Coordinate(Iter[2], Iter[5], Pads[0], Dilation[0], Strides[0], Source.Shape[2], InsideY);
  Z := Coordinate(Iter[3], Iter[6], Pads[1], Dilation[1], Strides[1], Source.Shape[3], InsideZ);
  Pixel := ReadTensor(X, [Iter[0], Channel, Y, Z]);
  Pixel := Intrinsic('tirx.if_then_else', FFloatType,
    [Primitive('tirx.And', InsideY, InsideZ), Pixel, FZero]);
  Product := Primitive('tirx.Mul', Pixel, ReadTensor(W,
    [Channel, Primitive('tirx.FloorMod', Iter[1], IntImm(Weights.Shape[1])), Iter[5], Iter[6]]));
  Compute := ComputeTensor('conv_transpose', ReduceSum(Product, [Axes[4], Axes[5], Axes[6]]),
    [Axes[0], Axes[1], Axes[2], Axes[3]]);
  Convolved := Emit([Source, Weights], [X, W], Compute, Shape, OnnxFloat, 'conv_transpose', 2);
  if (Length(Node.Inputs) > 2) and (Node.Inputs[2] <> '') then
  begin
    Bias := Input(Node, 2);
    RequireCpu((Length(Bias.Shape) = 1) and (Bias.Shape[0] = Shape[1]),
      'Invalid ONNX ConvTranspose bias');
    X := Placeholder(Shape, 'convolution');
    B := Placeholder(Bias.Shape, 'bias');
    Exit(Emit([Convolved, Bias], [X, B],
      Make('topi.nn.bias_add', [X, B, TTVMRuntime.AnyInt(1)]), Shape));
  end;
  Result := Convolved;
end;

function TOnnxBuilder.PatchConv3D(const Node: TOnnxNode;
  const Image, Weights: TGraphValue): TGraphValue;
var
  Pads, Strides, Dilation, Shape: TArray<Int64>;
  Patches, Filters, Bias, Matrix: TGraphValue;
  X, W, Row, Column, K, Rows, Columns, Inner, Compute, BiasTensor: TTVMFFIAny;
  Features, Count, Channels: Int64;
  I: Integer;
begin
  RequireCpu((Length(Weights.Shape) = 5) and (Image.Shape[1] = Weights.Shape[1]) and
    (Image.DataType = OnnxFloat) and (Weights.DataType = OnnxFloat) and
    (IntAttribute(Node, 'group', 1) = 1), 'Invalid ONNX 3D patch projection');
  Pads := IntsAttribute(Node, 'pads', [0, 0, 0, 0, 0, 0]);
  Strides := IntsAttribute(Node, 'strides', [1, 1, 1]);
  Dilation := IntsAttribute(Node, 'dilations', [1, 1, 1]);
  RequireCpu((Length(Pads) = 6) and (Length(Strides) = 3) and (Length(Dilation) = 3),
    'Invalid ONNX 3D patch attributes');
  for I := 0 to 2 do
    RequireCpu((Pads[I] = 0) and (Pads[I + 3] = 0) and (Dilation[I] = 1) and
      (Strides[I] = Weights.Shape[I + 2]) and (Image.Shape[I + 2] = Weights.Shape[I + 2]),
      'Only full 3D patch projection is supported');
  Count := Image.Shape[0];
  Channels := Weights.Shape[0];
  Features := TensorElements(Copy(Image.Shape, 1, 4));
  Patches := EmitReshape(Image, [Count, Features]);
  Filters := EmitReshape(Weights, [Channels, Features]);
  Shape := [Count, Channels];
  X := Placeholder(Patches.Shape, 'patches');
  W := Placeholder(Filters.Shape, 'filters');
  Rows := Iteration('patch', Count, False, Row);
  Columns := Iteration('channel', Channels, False, Column);
  Inner := Iteration('feature', Features, True, K);
  Compute := ComputeTensor('patch_projection',
    ReduceSum(Primitive('tirx.Mul', ReadTensor(X, [Row, K]), ReadTensor(W, [Column, K])), [Inner]),
    [Rows, Columns]);
  Matrix := Emit([Patches, Filters], [X, W], Compute, Shape, OnnxFloat, 'patch_projection', 1);
  if (Length(Node.Inputs) > 2) and (Node.Inputs[2] <> '') then
  begin
    Bias := Input(Node, 2);
    RequireCpu((Bias.DataType = OnnxFloat) and (Length(Bias.Shape) = 1) and
      (Bias.Shape[0] = Channels), 'Invalid ONNX 3D patch projection bias');
    X := Placeholder(Shape, 'projection');
    BiasTensor := Placeholder(Bias.Shape, 'bias');
    Compute := Make('topi.nn.bias_add', [X, BiasTensor, TTVMRuntime.AnyInt(1)]);
    Matrix := Emit([Matrix, Bias], [X, BiasTensor], Compute, Shape);
  end;
  Result := EmitReshape(Matrix, [Count, Channels, 1, 1, 1]);
end;

function TOnnxBuilder.Pool(const Node: TOnnxNode): TGraphValue;
var
  Source, Total: TGraphValue;
  InputTensor, Compute, Divisor: TTVMFFIAny;
  Shape, Kernel, Pads, Strides, Dilation: TArray<Int64>;
  CeilMode: Boolean;

  function OutputSize(const Dim, BeforePad, AfterPad, Window, Stride, Step: Int64): Int64;
  begin
    Result := SpatialSize(Dim, BeforePad, AfterPad, Window, Stride, Step);
    if CeilMode and ((Dim + BeforePad + AfterPad - Step * (Window - 1) - 1) mod Stride <> 0) and
      (Result * Stride < Dim + BeforePad) then Inc(Result);
  end;
begin
  Source := Input(Node, 0);
  RequireCpu(Length(Source.Shape) = 4, 'ONNX pooling needs NCHW input');
  CeilMode := IntAttribute(Node, 'ceil_mode', 0) <> 0;
  Shape := Copy(Source.Shape);
  InputTensor := Placeholder(Source.Shape, 'image');
  if Node.OpType = 'GlobalAveragePool' then
  begin
    RequireExplicitPadding(Node);
    Shape[2] := 1;
    Shape[3] := 1;
    if FCuda then
    begin
      Compute := Make('topi.sum', [InputTensor, IndexArray([2, 3]), TTVMRuntime.AnyBool(True)]);
      Total := Emit([Source], [InputTensor], Compute, Shape);
      InputTensor := Placeholder(Shape, 'total');
      Divisor := Make('ir.FloatImm', [TTVMRuntime.AnyDataType(FFloatDType),
        TTVMRuntime.AnyFloat(Source.Shape[2] * Source.Shape[3]), FEmpty]);
      Exit(Emit([Total], [InputTensor], Make('topi.divide', [InputTensor, Divisor]), Shape));
    end;
    Compute := Make('topi.nn.global_pool', [InputTensor, TTVMRuntime.AnyInt(0), TTVMRuntime.AnyCStr('NCHW')]);
    Exit(Emit([Source], [InputTensor], Compute, Shape));
  end;
  Kernel := IntsAttribute(Node, 'kernel_shape', [0, 0]);
  Strides := IntsAttribute(Node, 'strides', [1, 1]);
  Dilation := IntsAttribute(Node, 'dilations', [1, 1]);
  Pads := SpatialPads(Node, Source.Shape, Kernel, Strides, Dilation);
  Shape[2] := OutputSize(Shape[2], Pads[0], Pads[2], Kernel[0], Strides[0], Dilation[0]);
  Shape[3] := OutputSize(Shape[3], Pads[1], Pads[3], Kernel[1], Strides[1], Dilation[1]);
  Compute := Make('topi.nn.pool2d', [InputTensor, ShapeArray(Kernel), ShapeArray(Strides),
    ShapeArray(Dilation), ShapeArray(Pads), TTVMRuntime.AnyInt(Ord(Node.OpType = 'MaxPool')),
    TTVMRuntime.AnyBool(CeilMode), TTVMRuntime.AnyCStr('NCHW'),
    TTVMRuntime.AnyBool(IntAttribute(Node, 'count_include_pad', 0) <> 0)]);
  Result := Emit([Source], [InputTensor], Compute, Shape);
end;

function TOnnxBuilder.Concat(const Node: TOnnxNode): TGraphValue;
var
  Inputs: TArray<TGraphValue>;
  TensorInputs: TArray<TTVMFFIAny>;
  Shape, Values: TArray<Int64>;
  Axis, Inner, Outer, BlockSize, Group: Int64;
  I, J, Position: Integer;
  IsStatic: Boolean;
  Compute: TTVMFFIAny;
begin
  RequireCpu(Length(Node.Inputs) > 0, 'ONNX Concat has no inputs');
  SetLength(Inputs, Length(Node.Inputs));
  SetLength(TensorInputs, Length(Inputs));
  for I := 0 to High(Inputs) do Inputs[I] := Input(Node, I);
  Shape := Copy(Inputs[0].Shape);
  Axis := IntAttribute(Node, 'axis', 0);
  if Axis < 0 then Inc(Axis, Length(Shape));
  RequireCpu((Axis >= 0) and (Axis < Length(Shape)), 'ONNX Concat axis is out of range');
  Shape[Axis] := 0;
  IsStatic := Inputs[0].DataType = OnnxInt64;
  for I := 0 to High(Inputs) do
  begin
    RequireCpu((Length(Inputs[I].Shape) = Length(Shape)) and (Inputs[I].DataType = Inputs[0].DataType),
      'ONNX Concat input rank or type differs');
    for J := 0 to High(Shape) do
      if J <> Axis then
        RequireCpu(Inputs[I].Shape[J] = Shape[J], 'ONNX Concat input dimensions differ');
    Inc(Shape[Axis], Inputs[I].Shape[Axis]);
    IsStatic := IsStatic and Inputs[I].StaticInt;
  end;
  if IsStatic then
  begin
    SetLength(Values, TensorElements(Shape));
    Inner := TensorElements(Copy(Shape, Axis + 1, Length(Shape) - Axis - 1));
    Outer := TensorElements(Copy(Shape, 0, Axis));
    Position := 0;
    for Group := 0 to Outer - 1 do
      for I := 0 to High(Inputs) do
      begin
        BlockSize := Inputs[I].Shape[Axis] * Inner;
        if BlockSize > 0 then
          Move(Inputs[I].Integers[Group * BlockSize], Values[Position], BlockSize * SizeOf(Int64));
        Inc(Position, BlockSize);
      end;
    Exit(StaticIntegers(Shape, Values));
  end;
  for I := 0 to High(Inputs) do
    TensorInputs[I] := Placeholder(Inputs[I].Shape, 'input', Inputs[I].DataType);
  Compute := Make('topi.concatenate', [Make('ffi.Array', TensorInputs), TTVMRuntime.AnyInt(Axis)]);
  Result := Emit(Inputs, TensorInputs, Compute, Shape, Inputs[0].DataType);
end;

function TOnnxBuilder.Reshape(const Node: TOnnxNode): TGraphValue;
var
  Source: TGraphValue;
  Shape: TArray<Int64>;
  I, Inferred: Integer;
  Elements, Count: Int64;
begin
  Source := Input(Node, 0);
  RequireCpu(Length(Node.Inputs) = 2, 'ONNX Reshape needs a constant shape');
  Shape := StaticInput(Node, 1);
  Inferred := -1;
  for I := 0 to High(Shape) do
    if Shape[I] = -1 then
    begin
      RequireCpu(Inferred = -1, 'ONNX Reshape has more than one inferred dimension');
      Inferred := I;
      Shape[I] := 1;
    end
    else if Shape[I] = 0 then
      begin
        RequireCpu((IntAttribute(Node, 'allowzero', 0) = 0) and (I < Length(Source.Shape)),
          'Unsupported ONNX Reshape zero dimension');
        Shape[I] := Source.Shape[I];
      end;
  Elements := TensorElements(Source.Shape);
  Count := TensorElements(Shape);
  if Inferred >= 0 then
  begin
    RequireCpu(Elements mod Count = 0, 'ONNX Reshape dimensions do not divide the input size');
    Shape[Inferred] := Elements div Count;
  end;
  Result := EmitReshape(Source, Shape);
end;

function TOnnxBuilder.EmitReshape(const Source: TGraphValue; const Shape: TArray<Int64>): TGraphValue;
var
  InputTensor, Compute: TTVMFFIAny;
begin
  RequireCpu(TensorElements(Shape) = TensorElements(Source.Shape), 'ONNX Reshape changes the tensor size');
  if Source.StaticInt then Exit(StaticIntegers(Shape, Source.Integers, Source.DataType));
  if Source.StaticFloat then Exit(ConstantValue(Source.Floats, Shape));
  InputTensor := Placeholder(Source.Shape, 'input', Source.DataType);
  Compute := Make('topi.reshape', [InputTensor, ShapeArray(Shape)]);
  Result := Emit([Source], [InputTensor], Compute, Shape, Source.DataType);
end;

function TOnnxBuilder.Flatten(const Node: TOnnxNode): TGraphValue;
var
  Source: TGraphValue;
  Shape: TArray<Int64>;
  Axis: Int64;
begin
  Source := Input(Node, 0);
  Axis := IntAttribute(Node, 'axis', 1);
  if Axis < 0 then Inc(Axis, Length(Source.Shape));
  RequireCpu((Axis >= 0) and (Axis <= Length(Source.Shape)), 'ONNX Flatten axis is out of range');
  SetLength(Shape, 2);
  Shape[0] := TensorElements(Copy(Source.Shape, 0, Axis));
  Shape[1] := TensorElements(Copy(Source.Shape, Axis, Length(Source.Shape) - Axis));
  Result := EmitReshape(Source, Shape);
end;

function TOnnxBuilder.BatchNormalization(const Node: TOnnxNode): TGraphValue;
var
  Source, ScaleValue, BiasValue: TGraphValue;
  Scale, Bias, Mean, Variance: TArray<Single>;
  Shape: TArray<Int64>;
  InputTensor, ScaleTensor, BiasTensor, Compute: TTVMFFIAny;
  Epsilon: Single;
  I, Channels: Integer;

  function Parameter(const Index: Integer): TArray<Single>;
  var
    Tensor: TOnnxTensor;
  begin
    RequireCpu(FInitializers.TryGetValue(Node.Inputs[Index], Tensor) and
      (Length(Tensor.Shape) = 1) and (Tensor.Shape[0] = Channels),
      'ONNX BatchNormalization needs fixed channel parameters');
    Result := ReadOnnxFloat32(Tensor);
  end;

begin
  Source := Input(Node, 0);
  RequireCpu((Length(Node.Inputs) = 5) and (Length(Source.Shape) = 4) and
    (Source.Shape[1] <= High(Integer)) and (IntAttribute(Node, 'training_mode', 0) = 0),
    'Unsupported ONNX BatchNormalization signature');
  Channels := Source.Shape[1];
  Scale := Parameter(1);
  Bias := Parameter(2);
  Mean := Parameter(3);
  Variance := Parameter(4);
  Epsilon := FloatAttribute(Node, 'epsilon', 1E-5);
  RequireCpu(Epsilon > 0, 'Invalid ONNX BatchNormalization epsilon');
  for I := 0 to Channels - 1 do
  begin
    RequireCpu(Variance[I] >= 0, 'Invalid ONNX BatchNormalization variance');
    Scale[I] := Scale[I] / Sqrt(Variance[I] + Epsilon);
    Bias[I] := Bias[I] - Mean[I] * Scale[I];
  end;
  SetLength(Shape, 1);
  Shape[0] := Channels;
  ScaleValue := ConstantValue(Scale, Shape);
  BiasValue := ConstantValue(Bias, Shape);
  InputTensor := Placeholder(Source.Shape, 'input');
  ScaleTensor := Placeholder(Shape, 'scale');
  BiasTensor := Placeholder(Shape, 'bias');
  Compute := Make('topi.nn.scale_shift_nchw', [InputTensor, ScaleTensor, BiasTensor]);
  Result := Emit([Source, ScaleValue, BiasValue], [InputTensor, ScaleTensor, BiasTensor], Compute,
    Source.Shape);
end;

function BroadcastIndex(const SourceShape, Shape: TArray<Int64>; Position: Int64): Int64;
var
  Axis, SourceAxis: Integer;
  Index, Step: Int64;
begin
  Index := 0;
  Step := 1;
  for Axis := High(Shape) downto 0 do
  begin
    SourceAxis := Axis + Length(SourceShape) - Length(Shape);
    if SourceAxis < 0 then Break;
    if SourceShape[SourceAxis] > 1 then Index := Index + (Position mod Shape[Axis]) * Step;
    Step := Step * SourceShape[SourceAxis];
    Position := Position div Shape[Axis];
  end;
  Result := Index;
end;

function BroadcastInteger(const Value: TGraphValue; const Shape: TArray<Int64>; Position: Int64): Int64;
begin
  Result := Value.Integers[BroadcastIndex(Value.Shape, Shape, Position)];
end;

function TOnnxBuilder.Binary(const Node: TOnnxNode; const OperatorName: string): TGraphValue;
const
  StaticLimit = 4096;
var
  Left, Right: TGraphValue;
  Shape, Values: TArray<Int64>;
  LeftTensor, RightTensor, Compute: TTVMFFIAny;
  I, LeftIndex, RightIndex, Kind: Integer;
  LeftDim, RightDim, L, R: Int64;
  Floats: TArray<Single>;
  LF, RF: Single;
  Folded: Boolean;

begin
  Left := Input(Node, 0);
  Right := Input(Node, 1);
  RequireCpu(Left.DataType = Right.DataType, 'ONNX binary operand types differ');
  Kind := Left.DataType;
  if (Node.OpType = 'Less') or (Node.OpType = 'Greater') or
    (Node.OpType = 'Equal') or (Node.OpType = 'GreaterOrEqual') then Kind := OnnxBool;
  if Length(Left.Shape) > Length(Right.Shape) then
    SetLength(Shape, Length(Left.Shape))
  else
    SetLength(Shape, Length(Right.Shape));
  for I := 0 to High(Shape) do
  begin
    LeftIndex := I + Length(Left.Shape) - Length(Shape);
    RightIndex := I + Length(Right.Shape) - Length(Shape);
    LeftDim := 1;
    RightDim := 1;
    if LeftIndex >= 0 then LeftDim := Left.Shape[LeftIndex];
    if RightIndex >= 0 then RightDim := Right.Shape[RightIndex];
    RequireCpu((LeftDim = RightDim) or (LeftDim = 1) or (RightDim = 1),
      'ONNX binary dimensions cannot broadcast');
    if LeftDim > RightDim then
      Shape[I] := LeftDim
    else
      Shape[I] := RightDim;
  end;
  if Left.StaticInt and Right.StaticInt then
  begin
    SetLength(Values, TensorElements(Shape));
    for I := 0 to High(Values) do
    begin
      L := BroadcastInteger(Left, Shape, I);
      R := BroadcastInteger(Right, Shape, I);
      if Node.OpType = 'Add' then
        Values[I] := L + R
      else if Node.OpType = 'Sub' then
          Values[I] := L - R
        else if Node.OpType = 'Mul' then
            Values[I] := L * R
          else if Node.OpType = 'Div' then
            begin
              RequireCpu(R <> 0, 'ONNX integer division by zero');
              Values[I] := L div R;
            end
            else if Node.OpType = 'Mod' then
              begin
                RequireCpu((R <> 0) and not ((L = Low(Int64)) and (R = -1)),
                  'Invalid ONNX integer modulo');
                Values[I] := L mod R;
                if (IntAttribute(Node, 'fmod', 0) = 0) and (Values[I] <> 0) and
                  ((Values[I] < 0) <> (R < 0)) then Inc(Values[I], R);
              end
              else if Node.OpType = 'Max' then
                begin
                  Values[I] := L;
                  if R > L then Values[I] := R;
                end
                else if Node.OpType = 'Equal' then
                    Values[I] := Ord(L = R)
                  else if Node.OpType = 'Less' then
                      Values[I] := Ord(L < R)
                    else if Node.OpType = 'Greater' then
                        Values[I] := Ord(L > R)
                      else if Node.OpType = 'GreaterOrEqual' then
                          Values[I] := Ord(L >= R)
                        else if Node.OpType = 'And' then
                            Values[I] := Ord((L <> 0) and (R <> 0))
                          else
                            raise ECpuGraph.Create('Unsupported static ONNX integer operator: ' +
                              Node.OpType);
    end;
    Exit(StaticIntegers(Shape, Values, Kind));
  end;
  if Left.StaticFloat and Right.StaticFloat and (Kind = OnnxFloat) and
    (TensorElements(Shape) <= StaticLimit) and
    ((Node.OpType = 'Add') or (Node.OpType = 'Sub') or
    (Node.OpType = 'Mul') or (Node.OpType = 'Div')) then
    begin
      SetLength(Floats, TensorElements(Shape));
      Folded := True;
      for I := 0 to High(Floats) do
      begin
        LF := Left.Floats[BroadcastIndex(Left.Shape, Shape, I)];
        RF := Right.Floats[BroadcastIndex(Right.Shape, Shape, I)];
        if IsNan(LF) or IsInfinite(LF) or IsNan(RF) or IsInfinite(RF) or
          ((Node.OpType = 'Div') and (RF = 0)) then
          begin
            Folded := False;
            Break;
          end;
        if Node.OpType = 'Add' then
          Floats[I] := LF + RF
        else if Node.OpType = 'Sub' then
            Floats[I] := LF - RF
          else if Node.OpType = 'Mul' then
              Floats[I] := LF * RF
            else
              Floats[I] := LF / RF;
      end;
      if Folded then Exit(ConstantValue(Floats, Shape));
    end;
  RequireCpu(Node.OpType <> 'Mod', 'ONNX Mod requires constant integer operands');
  LeftTensor := Placeholder(Left.Shape, 'left', Left.DataType);
  RightTensor := Placeholder(Right.Shape, 'right', Right.DataType);
  Compute := Make(OperatorName, [LeftTensor, RightTensor]);
  Result := Emit([Left, Right], [LeftTensor, RightTensor], Compute, Shape, Kind);
end;

function TOnnxBuilder.Gemm(const Node: TOnnxNode): TGraphValue;
var
  A, B, Bias: TGraphValue;
  Shape: TArray<Int64>;
  LeftTensor, RightTensor, Weights, BiasTensor, Compute: TTVMFFIAny;
  Transpose: Boolean;
begin
  A := Input(Node, 0);
  B := Input(Node, 1);
  RequireCpu((Length(A.Shape) = 2) and (Length(B.Shape) = 2) and
    (IntAttribute(Node, 'transA', 0) = 0) and (FloatAttribute(Node, 'alpha', 1) = 1) and
    (FloatAttribute(Node, 'beta', 1) = 1), 'Unsupported ONNX Gemm signature');
  Transpose := IntAttribute(Node, 'transB', 0) = 0;
  SetLength(Shape, 2);
  Shape[0] := A.Shape[0];
  Shape[1] := B.Shape[Ord(Transpose)];
  RequireCpu(A.Shape[1] = B.Shape[Ord(not Transpose)], 'ONNX Gemm inner dimensions differ');
  LeftTensor := Placeholder(A.Shape, 'left');
  RightTensor := Placeholder(B.Shape, 'right');
  Weights := RightTensor;
  if Transpose then Weights := Make('topi.transpose', [RightTensor, IndexArray([1, 0])]);
  if (Length(Node.Inputs) > 2) and (Node.Inputs[2] <> '') then
  begin
    Bias := Input(Node, 2);
    RequireCpu((Length(Bias.Shape) = 1) and (Bias.Shape[0] = Shape[1]), 'Unsupported ONNX Gemm bias');
    BiasTensor := Placeholder(Bias.Shape, 'bias');
    Compute := Make('topi.nn.dense', [LeftTensor, Weights, BiasTensor, FFloatType]);
    Result := Emit([A, B, Bias], [LeftTensor, RightTensor, BiasTensor], Compute, Shape);
  end
  else begin
    Compute := Make('topi.nn.dense', [LeftTensor, Weights, FEmpty, FFloatType]);
    Result := Emit([A, B], [LeftTensor, RightTensor], Compute, Shape);
  end;
end;

function TOnnxBuilder.MatMul(const Node: TOnnxNode): TGraphValue;
var
  Left, Right: TGraphValue;
  Shape, RightPhysicalShape: TArray<Int64>;
  Iter, Axes, LeftIndices, RightIndices: TArray<TTVMFFIAny>;
  LeftTensor, RightTensor, LeftValue, RightValue, K, KAxis, Product, Compute: TTVMFFIAny;
  LeftBatch, RightBatch, Batch, I, Index, ParallelAxes: Integer;
  LeftDim, RightDim, Inner: Int64;
begin
  Left := Input(Node, 0);
  Right := Input(Node, 1);
  LeftBatch := Length(Left.Shape) - 2;
  RightBatch := Length(Right.Shape) - 2;
  RequireCpu((LeftBatch >= 0) and (RightBatch >= 0), 'ONNX MatMul requires tensors of rank two or more');
  Inner := Left.Shape[LeftBatch + 1];
  RequireCpu(Inner = Right.Shape[RightBatch], 'ONNX MatMul inner dimensions differ');
  Batch := LeftBatch;
  if RightBatch > Batch then Batch := RightBatch;
  SetLength(Shape, Batch + 2);
  for I := 0 to Batch - 1 do
  begin
    LeftDim := 1;
    RightDim := 1;
    Index := I + LeftBatch - Batch;
    if Index >= 0 then LeftDim := Left.Shape[Index];
    Index := I + RightBatch - Batch;
    if Index >= 0 then RightDim := Right.Shape[Index];
    RequireCpu((LeftDim = RightDim) or (LeftDim = 1) or (RightDim = 1),
      'ONNX MatMul batches cannot broadcast');
    if LeftDim > RightDim then
      Shape[I] := LeftDim
    else
      Shape[I] := RightDim;
  end;
  Shape[Batch] := Left.Shape[LeftBatch];
  Shape[Batch + 1] := Right.Shape[RightBatch + 1];
  SetLength(Axes, Length(Shape));
  SetLength(Iter, Length(Shape));
  for I := 0 to High(Shape) do
    Axes[I] := Iteration(UTF8Encode('axis' + IntToStr(I)), Shape[I], False, Iter[I]);
  KAxis := Iteration('k', Inner, True, K);
  SetLength(LeftIndices, LeftBatch + 2);
  for I := 0 to LeftBatch - 1 do
    if Left.Shape[I] = 1 then
      LeftIndices[I] := IntImm(0)
    else
      LeftIndices[I] := Iter[I + Batch - LeftBatch];
  LeftIndices[LeftBatch] := Iter[Batch];
  LeftIndices[LeftBatch + 1] := K;
  SetLength(RightIndices, RightBatch + 2);
  for I := 0 to RightBatch - 1 do
    if Right.Shape[I] = 1 then
      RightIndices[I] := IntImm(0)
    else
      RightIndices[I] := Iter[I + Batch - RightBatch];
  RightIndices[RightBatch] := K;
  RightIndices[RightBatch + 1] := Iter[Batch + 1];
  RightPhysicalShape := Right.Shape;
  if Right.Transposed then
  begin
    RightPhysicalShape := Copy(Right.Shape);
    RightPhysicalShape[RightBatch] := Right.Shape[RightBatch + 1];
    RightPhysicalShape[RightBatch + 1] := Right.Shape[RightBatch];
    RightIndices[RightBatch] := Iter[Batch + 1];
    RightIndices[RightBatch + 1] := K;
  end;
  LeftTensor := Placeholder(Left.Shape, 'left');
  RightTensor := Placeholder(RightPhysicalShape, 'right');
  LeftValue := ReadTensor(LeftTensor, LeftIndices);
  RightValue := ReadTensor(RightTensor, RightIndices);
  Product := Primitive('tirx.Mul', LeftValue, RightValue);
  Compute := ComputeTensor('matmul', ReduceSum(Product, [KAxis]), Axes);
  ParallelAxes := 0;
  if TensorElements(Shape) >= (2 * 1024 * 1024 + Inner - 1) div Inner then
  begin
    ParallelAxes := Length(Shape);
    if TensorElements(Shape) div Shape[High(Shape)] >= 16 then Dec(ParallelAxes);
  end;
  Result := Emit([Left, Right], [LeftTensor, RightTensor], Compute, Shape, OnnxFloat, 'matmul', ParallelAxes);
end;

function TOnnxBuilder.Q8MatMul(const Node: TOnnxNode): TGraphValue;
var
  Source, Weights, Scales: TGraphValue;
  Shape: TArray<Int64>;
  Iter, Axes, SourceIndices: TArray<TTVMFFIAny>;
  X, Q, Scale, Column, Axis, Value, Coefficient, Factor, Product, Compute: TTVMFFIAny;
  I, Last: Integer;
begin
  Source := Input(Node, 0);
  Weights := Input(Node, 1);
  Scales := Input(Node, 2);
  Last := High(Source.Shape);
  RequireCpu((Last >= 1) and (Source.DataType = OnnxFloat) and
    (Weights.DataType = OnnxInt8) and (Length(Weights.Shape) = 2) and
    (Scales.DataType = OnnxFloat) and (Length(Scales.Shape) = 1) and
    (Source.Shape[Last] = Weights.Shape[1]) and (Scales.Shape[0] = Weights.Shape[0]),
    'Invalid int8 matrix multiplication dimensions or types');
  Shape := Copy(Source.Shape);
  Shape[Last] := Weights.Shape[0];
  SetLength(Iter, Length(Shape));
  SetLength(Axes, Length(Shape));
  for I := 0 to Last do
    Axes[I] := Iteration(UTF8Encode('axis' + IntToStr(I)), Shape[I], False, Iter[I]);
  Axis := Iteration('column', Source.Shape[Last], True, Column);
  SourceIndices := Copy(Iter);
  SourceIndices[Last] := Column;
  X := Placeholder(Source.Shape, 'input');
  Q := Placeholder(Weights.Shape, 'weights', OnnxInt8);
  Scale := Placeholder(Scales.Shape, 'scales');
  Value := ReadTensor(X, SourceIndices);
  Coefficient := Make('tirx.Cast', [FFloatType, ReadTensor(Q, [Iter[Last], Column]), FEmpty]);
  Factor := ReadTensor(Scale, [Iter[Last]]);
  Product := Primitive('tirx.Mul', Primitive('tirx.Mul', Value, Coefficient), Factor);
  Compute := ComputeTensor('q8_dot', ReduceSum(Product, [Axis]), Axes);
  Result := Emit([Source, Weights, Scales], [X, Q, Scale], Compute, Shape, OnnxFloat, 'q8_dot',
    Length(Shape), True);
end;

function TOnnxBuilder.Unary(const Node: TOnnxNode; const OperatorName: string): TGraphValue;
const
  Float32IntegralLimit = 8388608;
var
  Source: TGraphValue;
  InputTensor, Compute: TTVMFFIAny;
  Axes, Iter: TArray<TTVMFFIAny>;
  Floats: TArray<Single>;
  Value: Single;
  I: Integer;
  Folded: Boolean;
begin
  Source := Input(Node, 0);
  if Source.StaticFloat and ((Node.OpType = 'Ceil') or (Node.OpType = 'Floor')) then
  begin
    Floats := Copy(Source.Floats);
    Folded := True;
    for I := 0 to High(Floats) do
    begin
      Value := Floats[I];
      if IsNan(Value) or IsInfinite(Value) then
      begin
        Folded := False;
        Break;
      end;
      if Abs(Value) >= Float32IntegralLimit then Continue;
      if Node.OpType = 'Ceil' then
        Floats[I] := Ceil(Value)
      else
        Floats[I] := Floor(Value);
      if Floats[I] = 0 then Floats[I] := Value * 0;
    end;
    if Folded then Exit(ConstantValue(Floats, Source.Shape));
  end;
  InputTensor := Placeholder(Source.Shape, 'input', Source.DataType);
  if (Node.OpType = 'Ceil') or (Node.OpType = 'Floor') then
  begin
    RequireCpu(Source.DataType = OnnxFloat, 'ONNX rounding requires a float32 tensor');
    SetLength(Axes, Length(Source.Shape));
    SetLength(Iter, Length(Source.Shape));
    for I := 0 to High(Axes) do
      Axes[I] := Iteration(UTF8Encode('axis' + IntToStr(I)), Source.Shape[I], False, Iter[I]);
    if Node.OpType = 'Ceil' then
      Compute := Intrinsic('tirx.ceil', FFloatType, [ReadTensor(InputTensor, Iter)])
    else
      Compute := Intrinsic('tirx.floor', FFloatType, [ReadTensor(InputTensor, Iter)]);
    Compute := ComputeTensor('round', Compute, Axes);
    Exit(Emit([Source], [InputTensor], Compute, Source.Shape));
  end;
  Result := Emit([Source], [InputTensor], Make(OperatorName, [InputTensor]), Source.Shape, Source.DataType);
end;

function TOnnxBuilder.Range(const Node: TOnnxNode): TGraphValue;
const
  ElementLimit = 1024 * 1024;
var
  First, Last, Delta: TGraphValue;
  Values: TArray<Int64>;
  Floats: TArray<Single>;
  Start, Limit, Step, Current: Int64;
  FloatStart, FloatLimit, FloatStep, FloatCount: Double;
  Count, I: Integer;
begin
  RequireCpu(Length(Node.Inputs) = 3, 'ONNX Range requires three inputs');
  First := Input(Node, 0);
  Last := Input(Node, 1);
  Delta := Input(Node, 2);
  RequireCpu((Length(First.Shape) = 0) and (Length(Last.Shape) = 0) and
    (Length(Delta.Shape) = 0), 'ONNX Range requires scalar inputs');
  if First.StaticFloat and Last.StaticFloat and Delta.StaticFloat then
  begin
    FloatStart := First.Floats[0];
    FloatLimit := Last.Floats[0];
    FloatStep := Delta.Floats[0];
    RequireCpu(not IsNan(FloatStart) and not IsInfinite(FloatStart) and
      not IsNan(FloatLimit) and not IsInfinite(FloatLimit) and
      not IsNan(FloatStep) and not IsInfinite(FloatStep), 'ONNX Range requires finite constants');
    RequireCpu(FloatStep <> 0, 'ONNX Range step cannot be zero');
    FloatCount := (FloatLimit - FloatStart) / FloatStep;
    RequireCpu(FloatCount > 0, 'Empty ONNX Range is not supported');
    RequireCpu(FloatCount <= ElementLimit, 'ONNX Range exceeds the constant element limit');
    Count := Ceil(FloatCount);
    SetLength(Floats, Count);
    for I := 0 to High(Floats) do Floats[I] := FloatStart + I * FloatStep;
    Exit(ConstantValue(Floats, [Count]));
  end;
  RequireCpu(First.StaticInt and Last.StaticInt and Delta.StaticInt and
    (First.DataType = OnnxInt64) and (Last.DataType = OnnxInt64) and
    (Delta.DataType = OnnxInt64) and (Length(First.Shape) = 0) and
    (Length(Last.Shape) = 0) and (Length(Delta.Shape) = 0),
    'ONNX Range requires constant int64 scalars');
  Start := First.Integers[0];
  Limit := Last.Integers[0];
  Step := Delta.Integers[0];
  RequireCpu(Step <> 0, 'ONNX Range step cannot be zero');
  SetLength(Values, ElementLimit);
  Current := Start;
  Count := 0;
  while ((Step > 0) and (Current < Limit)) or ((Step < 0) and (Current > Limit)) do
  begin
    RequireCpu(Count < ElementLimit, 'ONNX Range exceeds the constant element limit');
    Values[Count] := Current;
    Inc(Count);
    if Step > 0 then
    begin
      if Current > High(Int64) - Step then Break;
    end
    else if Current < Low(Int64) - Step then
        Break;
    Inc(Current, Step);
  end;
  RequireCpu(Count > 0, 'Empty ONNX Range is not supported');
  SetLength(Values, Count);
  Result := StaticIntegers([Count], Values);
end;

function TOnnxBuilder.ShapeOf(const Node: TOnnxNode): TGraphValue;
var
  Source: TGraphValue;
  First, Last, Rank: Int64;

  function Bound(const Value: Int64): Int64;
  begin
    Result := Value;
    if Result < 0 then Inc(Result, Rank);
    if Result < 0 then Exit(0);
    if Result > Rank then Result := Rank;
  end;

begin
  Source := Input(Node, 0);
  Rank := Length(Source.Shape);
  First := Bound(IntAttribute(Node, 'start', 0));
  Last := Bound(IntAttribute(Node, 'end', Rank));
  if Last < First then Last := First;
  Result := StaticIntegers([Last - First], Copy(Source.Shape, First, Last - First));
end;

function TOnnxBuilder.Unsqueeze(const Node: TOnnxNode): TGraphValue;
var
  Source: TGraphValue;
  Axes, Shape: TArray<Int64>;
  Inserted: TArray<Boolean>;
  Axis: Int64;
  I, J, Rank: Integer;
begin
  Source := Input(Node, 0);
  if FOpset >= 13 then
    Axes := StaticInput(Node, 1)
  else
    Axes := IntsAttribute(Node, 'axes', []);
  Rank := Length(Source.Shape) + Length(Axes);
  SetLength(Shape, Rank);
  SetLength(Inserted, Rank);
  for I := 0 to High(Axes) do
  begin
    Axis := Axes[I];
    if Axis < 0 then Inc(Axis, Rank);
    RequireCpu((Axis >= 0) and (Axis < Rank) and not Inserted[Axis], 'Invalid ONNX Unsqueeze axis');
    Inserted[Axis] := True;
  end;
  J := 0;
  for I := 0 to High(Shape) do
    if Inserted[I] then
      Shape[I] := 1
    else begin
      Shape[I] := Source.Shape[J];
      Inc(J);
    end;
  Result := EmitReshape(Source, Shape);
end;

function TOnnxBuilder.Squeeze(const Node: TOnnxNode): TGraphValue;
var
  Source: TGraphValue;
  Axes, Shape: TArray<Int64>;
  Removed: TArray<Boolean>;
  Axis: Int64;
  I, Count, Rank: Integer;
begin
  Source := Input(Node, 0);
  Rank := Length(Source.Shape);
  if (Length(Node.Inputs) > 1) and (Node.Inputs[1] <> '') then
    Axes := StaticInput(Node, 1)
  else
    Axes := IntsAttribute(Node, 'axes', []);
  SetLength(Removed, Rank);
  if Length(Axes) = 0 then
  begin
    for I := 0 to Rank - 1 do Removed[I] := Source.Shape[I] = 1;
  end
  else
    for I := 0 to High(Axes) do
    begin
      Axis := Axes[I];
      if Axis < 0 then Inc(Axis, Rank);
      RequireCpu((Axis >= 0) and (Axis < Rank) and not Removed[Axis] and
        (Source.Shape[Axis] = 1), 'Invalid ONNX Squeeze axis');
      Removed[Axis] := True;
    end;
  SetLength(Shape, Rank);
  Count := 0;
  for I := 0 to Rank - 1 do
    if not Removed[I] then
    begin
      Shape[Count] := Source.Shape[I];
      Inc(Count);
    end;
  SetLength(Shape, Count);
  Result := EmitReshape(Source, Shape);
end;

function TOnnxBuilder.Gather(const Node: TOnnxNode; const Elementwise: Boolean): TGraphValue;
var
  Source, Indices: TGraphValue;
  InputTensor, IndexTensor, Compute: TTVMFFIAny;
  Shape, Values: TArray<Int64>;
  Axis, Index: Int64;
  I: Integer;
begin
  Source := Input(Node, 0);
  Indices := Input(Node, 1);
  Axis := IntAttribute(Node, 'axis', 0);
  if Axis < 0 then Inc(Axis, Length(Source.Shape));
  RequireCpu((Axis >= 0) and (Axis < Length(Source.Shape)) and (Indices.DataType = OnnxInt64),
    'Invalid ONNX Gather axis or index type');
  if Elementwise then
  begin
    RequireCpu(Length(Source.Shape) = Length(Indices.Shape), 'ONNX GatherElements ranks differ');
    for I := 0 to High(Source.Shape) do
      if I <> Axis then
        RequireCpu(Indices.Shape[I] <= Source.Shape[I], 'ONNX GatherElements dimensions differ');
    Shape := Indices.Shape;
  end
  else begin
    SetLength(Shape, Length(Source.Shape) + Length(Indices.Shape) - 1);
    for I := 0 to Axis - 1 do Shape[I] := Source.Shape[I];
    for I := 0 to High(Indices.Shape) do Shape[Axis + I] := Indices.Shape[I];
    for I := Axis + 1 to High(Source.Shape) do
      Shape[I + Length(Indices.Shape) - 1] := Source.Shape[I];
  end;
  if not Elementwise and Source.StaticInt and Indices.StaticInt and (Length(Source.Shape) = 1) then
  begin
    SetLength(Values, Length(Indices.Integers));
    for I := 0 to High(Values) do
    begin
      Index := Indices.Integers[I];
      if Index < 0 then Inc(Index, Source.Shape[0]);
      RequireCpu((Index >= 0) and (Index < Source.Shape[0]), 'ONNX Gather index is out of range');
      Values[I] := Source.Integers[Index];
    end;
    Exit(StaticIntegers(Shape, Values, Source.DataType));
  end;
  InputTensor := Placeholder(Source.Shape, 'input', Source.DataType);
  IndexTensor := Placeholder(Indices.Shape, 'indices', Indices.DataType);
  if Elementwise then
    Compute := Make('topi.gather', [InputTensor, TTVMRuntime.AnyInt(Axis), IndexTensor])
  else
    Compute := Make('topi.take', [InputTensor, IndexTensor, TTVMRuntime.AnyInt(0),
      TTVMRuntime.AnyInt(Axis), TTVMRuntime.AnyCStr('wrap')]);
  Result := Emit([Source, Indices], [InputTensor, IndexTensor], Compute, Shape, Source.DataType);
end;

function TOnnxBuilder.Reduce(const Node: TOnnxNode): TGraphValue;
var
  Source, Total: TGraphValue;
  Axes, Shape: TArray<Int64>;
  Reduced: TArray<Boolean>;
  InputTensor, Compute, Divisor: TTVMFFIAny;
  KeepDims: Boolean;
  I, Count: Integer;
  Axis, Elements: Int64;
begin
  Source := Input(Node, 0);
  if (Length(Node.Inputs) > 1) and (Node.Inputs[1] <> '') then
    Axes := StaticInput(Node, 1)
  else
    Axes := IntsAttribute(Node, 'axes', []);
  if Length(Axes) = 0 then
  begin
    if IntAttribute(Node, 'noop_with_empty_axes', 0) <> 0 then Exit(Source);
    SetLength(Axes, Length(Source.Shape));
    for I := 0 to High(Axes) do Axes[I] := I;
  end;
  if Length(Axes) = 0 then Exit(Source);
  SetLength(Reduced, Length(Source.Shape));
  Elements := 1;
  for I := 0 to High(Axes) do
  begin
    Axis := Axes[I];
    if Axis < 0 then Inc(Axis, Length(Source.Shape));
    RequireCpu((Axis >= 0) and (Axis < Length(Source.Shape)) and not Reduced[Axis],
      'Invalid ONNX reduction axis');
    Axes[I] := Axis;
    Reduced[Axis] := True;
    Elements := Elements * Source.Shape[Axis];
  end;
  KeepDims := IntAttribute(Node, 'keepdims', 1) <> 0;
  SetLength(Shape, Length(Source.Shape));
  Count := 0;
  for I := 0 to High(Shape) do
  begin
    if Reduced[I] and not KeepDims then Continue;
    if Reduced[I] then
      Shape[Count] := 1
    else
      Shape[Count] := Source.Shape[I];
    Inc(Count);
  end;
  SetLength(Shape, Count);
  InputTensor := Placeholder(Source.Shape, 'input', Source.DataType);
  if Node.OpType = 'ReduceMax' then
    Compute := Make('topi.max', [InputTensor, IndexArray(Axes), TTVMRuntime.AnyBool(KeepDims)])
  else
    Compute := Make('topi.sum', [InputTensor, IndexArray(Axes), TTVMRuntime.AnyBool(KeepDims)]);
  if Node.OpType = 'ReduceMean' then
  begin
    RequireCpu(Source.DataType = OnnxFloat, 'ONNX ReduceMean requires FP32');
    Divisor := Make('ir.FloatImm',
      [TTVMRuntime.AnyDataType(FFloatDType), TTVMRuntime.AnyFloat(Elements), FEmpty]);
    if FCuda then
    begin
      Total := Emit([Source], [InputTensor], Compute, Shape);
      InputTensor := Placeholder(Shape, 'total');
      Exit(Emit([Total], [InputTensor], Make('topi.divide', [InputTensor, Divisor]), Shape));
    end;
    Compute := Make('topi.divide', [Compute, Divisor]);
  end;
  Result := Emit([Source], [InputTensor], Compute, Shape, Source.DataType);
end;

function TOnnxBuilder.Pad(const Node: TOnnxNode): TGraphValue;
var
  Source: TGraphValue;
  Pads, Shape: TArray<Int64>;
  ConstantTensor: TOnnxTensor;
  ConstantValues: TArray<Single>;
  InputTensor, Compute, PadValue: TTVMFFIAny;
  I, Rank: Integer;
begin
  Source := Input(Node, 0);
  Rank := Length(Source.Shape);
  RequireCpu((Source.DataType = OnnxFloat) and (Length(Node.Inputs) in [2, 3]) and (StringAttribute(Node, 'mode', 'constant') = 'constant'),
    'Unsupported ONNX Pad mode or inputs');
  Pads := StaticInput(Node, 1);
  RequireCpu(Length(Pads) = 2 * Rank, 'ONNX Pad rank differs');
  PadValue := FZero;
  if (Length(Node.Inputs) = 3) and (Node.Inputs[2] <> '') then
  begin
    RequireCpu(FInitializers.TryGetValue(Node.Inputs[2], ConstantTensor), 'ONNX Pad value must be constant');
    ConstantValues := ReadOnnxFloat32(ConstantTensor);
    RequireCpu(Length(ConstantValues) = 1, 'ONNX Pad value must be scalar');
    PadValue := Make('ir.FloatImm',
      [TTVMRuntime.AnyDataType(FFloatDType), TTVMRuntime.AnyFloat(ConstantValues[0]), FEmpty]);
  end;
  Shape := Copy(Source.Shape);
  for I := 0 to High(Shape) do
  begin
    RequireCpu((Pads[I] >= 0) and (Pads[I + Rank] >= 0), 'ONNX Pad requires nonnegative widths');
    Shape[I] := Shape[I] + Pads[I] + Pads[I + Rank];
  end;
  InputTensor := Placeholder(Source.Shape, 'input');
  Compute := Make('topi.nn.pad', [InputTensor, ShapeArray(Copy(Pads, 0, Rank)),
    ShapeArray(Copy(Pads, Rank, Rank)), PadValue]);
  Result := Emit([Source], [InputTensor], Compute, Shape);
end;

function TOnnxBuilder.Cast(const Node: TOnnxNode): TGraphValue;
var
  Source: TGraphValue;
  Integers: TArray<Int64>;
  Floats: TArray<Single>;
  InputTensor, Compute: TTVMFFIAny;
  Kind, I: Integer;
begin
  Source := Input(Node, 0);
  Kind := IntAttribute(Node, 'to', 0);
  if Kind = Source.DataType then Exit(Source);
  if Source.StaticFloat and (Kind = OnnxInt64) then
  begin
    SetLength(Integers, Length(Source.Floats));
    for I := 0 to High(Integers) do
    begin
      RequireCpu(not IsNan(Source.Floats[I]) and not IsInfinite(Source.Floats[I]) and
        (Source.Floats[I] >= -9223372036854775808.0) and
        (Source.Floats[I] < 9223372036854775808.0), 'ONNX Cast constant is outside int64 range');
      Integers[I] := Trunc(Source.Floats[I]);
    end;
    Exit(StaticIntegers(Source.Shape, Integers));
  end;
  if Source.StaticInt then
    case Kind of
      OnnxFloat:
        begin
          SetLength(Floats, Length(Source.Integers));
          for I := 0 to High(Floats) do Floats[I] := Source.Integers[I];
          Exit(ConstantValue(Floats, Source.Shape));
        end;
      OnnxBool, OnnxInt64:
        begin
          Integers := Copy(Source.Integers);
          if Kind = OnnxBool then
            for I := 0 to High(Integers) do Integers[I] := Ord(Integers[I] <> 0);
          Exit(StaticIntegers(Source.Shape, Integers, Kind));
        end;
    end;
  InputTensor := Placeholder(Source.Shape, 'input', Source.DataType);
  Compute := Make('topi.cast', [InputTensor, PrimitiveType(Kind)]);
  Result := Emit([Source], [InputTensor], Compute, Source.Shape, Kind);
end;

function TOnnxBuilder.ExpandOrTile(const Node: TOnnxNode): TGraphValue;
const
  FloatElementLimit = 65536;
var
  Source: TGraphValue;
  Spec, Shape, Values: TArray<Int64>;
  Floats: TArray<Single>;
  InputTensor, Compute: TTVMFFIAny;
  I, Axis, Offset: Integer;
  Dim, SourceDim, Position, Index, Step: Int64;
begin
  Source := Input(Node, 0);
  Spec := StaticInput(Node, 1);
  if Node.OpType = 'Tile' then
  begin
    RequireCpu(Length(Spec) = Length(Source.Shape), 'ONNX Tile repeat count differs from rank');
    Shape := Copy(Source.Shape);
    for I := 0 to High(Shape) do
    begin
      RequireCpu((Spec[I] > 0) and (Shape[I] <= High(Int64) div Spec[I]), 'Invalid ONNX Tile repeat');
      Shape[I] := Shape[I] * Spec[I];
    end;
  end
  else begin
    SetLength(Shape, Length(Source.Shape));
    if Length(Spec) > Length(Shape) then SetLength(Shape, Length(Spec));
    for I := 0 to High(Shape) do
    begin
      Dim := 1;
      Axis := I + Length(Spec) - Length(Shape);
      if Axis >= 0 then Dim := Spec[Axis];
      SourceDim := 1;
      Axis := I + Length(Source.Shape) - Length(Shape);
      if Axis >= 0 then SourceDim := Source.Shape[Axis];
      RequireCpu((Dim > 0) and ((Dim = SourceDim) or (Dim = 1) or (SourceDim = 1)),
        'Invalid ONNX Expand dimensions');
      if Dim < SourceDim then Dim := SourceDim;
      Shape[I] := Dim;
    end;
  end;
  if Source.StaticInt or (Source.StaticFloat and (TensorElements(Shape) <= FloatElementLimit)) then
  begin
    if Source.StaticInt then
      SetLength(Values, TensorElements(Shape))
    else
      SetLength(Floats, TensorElements(Shape));
    Offset := Length(Shape) - Length(Source.Shape);
    for I := 0 to TensorElements(Shape) - 1 do
    begin
      Position := I;
      Index := 0;
      Step := 1;
      for Axis := High(Shape) downto Offset do
      begin
        Dim := Source.Shape[Axis - Offset];
        Index := Index + ((Position mod Shape[Axis]) mod Dim) * Step;
        Position := Position div Shape[Axis];
        Step := Step * Dim;
      end;
      if Source.StaticInt then
        Values[I] := Source.Integers[Index]
      else
        Floats[I] := Source.Floats[Index];
    end;
    if Source.StaticInt then Exit(StaticIntegers(Shape, Values, Source.DataType));
    Exit(ConstantValue(Floats, Shape));
  end;
  InputTensor := Placeholder(Source.Shape, 'input', Source.DataType);
  if Node.OpType = 'Tile' then
    Compute := Make('topi.tile', [InputTensor, IndexArray(Spec)])
  else
    Compute := Make('topi.broadcast_to', [InputTensor, ShapeArray(Shape)]);
  Result := Emit([Source], [InputTensor], Compute, Shape, Source.DataType);
end;

function TOnnxBuilder.ConstantOfShape(const Node: TOnnxNode): TGraphValue;
var
  Attribute: TOnnxAttribute;
  Shape, Integers, IntValue: TArray<Int64>;
  Floats, FloatValue: TArray<Single>;
  Count: Int64;
  I: Integer;
begin
  Shape := StaticInput(Node, 0);
  Count := TensorElements(Shape);
  RequireCpu(Count <= High(Integer) div SizeOf(Int64), 'ONNX ConstantOfShape tensor is too large');
  for Attribute in Node.Attributes do
    if Attribute.Name = 'value' then
    begin
      if Attribute.Tensor.DataType = OnnxInt64 then
      begin
        IntValue := ReadOnnxInt64(Attribute.Tensor);
        RequireCpu(Length(IntValue) = 1, 'ONNX ConstantOfShape requires one value');
        SetLength(Integers, Count);
        for I := 0 to High(Integers) do Integers[I] := IntValue[0];
        Exit(StaticIntegers(Shape, Integers));
      end;
      FloatValue := ReadOnnxFloat32(Attribute.Tensor);
      RequireCpu(Length(FloatValue) = 1, 'ONNX ConstantOfShape requires one value');
      Break;
    end;
  SetLength(Floats, Count);
  if Length(FloatValue) > 0 then
    for I := 0 to High(Floats) do Floats[I] := FloatValue[0];
  Result := ConstantValue(Floats, Shape);
end;

function TOnnxBuilder.LayerNormalization(const Node: TOnnxNode): TGraphValue;
var
  Source, Scale, Bias: TGraphValue;
  InputTensor, ScaleTensor, BiasTensor, Compute: TTVMFFIAny;
  Axis: Int64;
  Axes: TArray<Int64>;
  I: Integer;
begin
  Source := Input(Node, 0);
  Scale := Input(Node, 1);
  Bias := Input(Node, 2);
  Axis := IntAttribute(Node, 'axis', -1);
  if Axis < 0 then Inc(Axis, Length(Source.Shape));
  RequireCpu(
    (Axis >= 0) and (Axis < Length(Source.Shape)) and (Source.DataType = OnnxFloat) and (Scale.DataType = OnnxFloat) and (Bias.DataType = OnnxFloat) and (Length(Scale.Shape) = Length(Source.Shape) - Axis) and (Length(Bias.Shape) = Length(Scale.Shape)),
    'Unsupported ONNX LayerNormalization types or axes'
  );
  SetLength(Axes, Length(Scale.Shape));
  for I := 0 to High(Axes) do
  begin
    Axes[I] := Axis + I;
    RequireCpu((Scale.Shape[I] = Source.Shape[Axes[I]]) and (Bias.Shape[I] = Scale.Shape[I]),
      'ONNX LayerNormalization parameter dimensions differ');
  end;
  InputTensor := Placeholder(Source.Shape, 'input');
  ScaleTensor := Placeholder(Scale.Shape, 'scale');
  BiasTensor := Placeholder(Bias.Shape, 'bias');
  Compute := Make('topi.nn.layer_norm', [InputTensor, ScaleTensor, BiasTensor, IndexArray(Axes),
    TTVMRuntime.AnyFloat(FloatAttribute(Node, 'epsilon', 1e-5))]);
  Result := Emit([Source, Scale, Bias], [InputTensor, ScaleTensor, BiasTensor], Compute, Source.Shape);
end;

function TOnnxBuilder.Clip(const Node: TOnnxNode): TGraphValue;
var
  Source: TGraphValue;
  InputTensor, Compute: TTVMFFIAny;

  function Bound(const Index: Integer; const Minimum: Boolean): TTVMFFIAny;
  var
    Tensor: TOnnxTensor;
    Values: TArray<Single>;
    Integers: TArray<Int64>;
    Value: Single;
    HasBound: Boolean;
  begin
    HasBound := (Index < Length(Node.Inputs)) and (Node.Inputs[Index] <> '');
    if Source.DataType = OnnxInt64 then
    begin
      if not HasBound then
      begin
        if Minimum then Exit(IntImm(Low(Int64)));
        Exit(IntImm(High(Int64)));
      end;
      Integers := StaticInput(Node, Index);
      RequireCpu(Length(Integers) = 1, 'ONNX Clip bound must be scalar');
      Exit(IntImm(Integers[0]));
    end;
    Value := 3.402823466e38;
    if Minimum then Value := -Value;
    if HasBound then
    begin
      RequireCpu(FInitializers.TryGetValue(Node.Inputs[Index], Tensor), 'ONNX Clip needs a constant bound');
      Values := ReadOnnxFloat32(Tensor);
      RequireCpu(Length(Values) = 1, 'ONNX Clip bound must be scalar');
      Value := Values[0];
    end;
    Result := Make('ir.FloatImm',
      [TTVMRuntime.AnyDataType(FFloatDType), TTVMRuntime.AnyFloat(Value), FEmpty]);
  end;

begin
  Source := Input(Node, 0);
  RequireCpu((FOpset >= 11) and (Source.DataType in [OnnxFloat, OnnxInt64]),
    'Unsupported ONNX Clip type or opset');
  InputTensor := Placeholder(Source.Shape, 'input', Source.DataType);
  Compute := Make('topi.clip', [InputTensor, Bound(1, True), Bound(2, False)]);
  Result := Emit([Source], [InputTensor], Compute, Source.Shape, Source.DataType);
end;

function TOnnxBuilder.Where(const Node: TOnnxNode): TGraphValue;
var
  Inputs: array[0..2] of TGraphValue;
  Tensors: array[0..2] of TTVMFFIAny;
  Shape, Values: TArray<Int64>;
  Compute: TTVMFFIAny;
  I, J, Rank, Axis: Integer;
  Dim: Int64;
begin
  Rank := 0;
  for I := 0 to High(Inputs) do
  begin
    Inputs[I] := Input(Node, I);
    if Length(Inputs[I].Shape) > Rank then Rank := Length(Inputs[I].Shape);
  end;
  RequireCpu((Inputs[0].DataType = OnnxBool) and (Inputs[1].DataType = Inputs[2].DataType),
    'Invalid ONNX Where input types');
  SetLength(Shape, Rank);
  for J := 0 to High(Shape) do Shape[J] := 1;
  for I := 0 to High(Inputs) do
  begin
    for J := 0 to High(Inputs[I].Shape) do
    begin
      Axis := Rank - Length(Inputs[I].Shape) + J;
      Dim := Inputs[I].Shape[J];
      RequireCpu((Dim = Shape[Axis]) or (Dim = 1) or (Shape[Axis] = 1),
        'ONNX Where dimensions cannot broadcast');
      if Dim > Shape[Axis] then Shape[Axis] := Dim;
    end;
  end;
  if Inputs[0].StaticInt and Inputs[1].StaticInt and Inputs[2].StaticInt then
  begin
    SetLength(Values, TensorElements(Shape));
    for I := 0 to High(Values) do
      if BroadcastInteger(Inputs[0], Shape, I) <> 0 then
        Values[I] := BroadcastInteger(Inputs[1], Shape, I)
      else
        Values[I] := BroadcastInteger(Inputs[2], Shape, I);
    Exit(StaticIntegers(Shape, Values, Inputs[1].DataType));
  end;
  for I := 0 to High(Tensors) do
    Tensors[I] := Placeholder(Inputs[I].Shape, 'input', Inputs[I].DataType);
  Compute := Make('topi.where', [Tensors[0], Tensors[1], Tensors[2]]);
  Result := Emit(Inputs, Tensors, Compute, Shape, Inputs[1].DataType);
end;

function TOnnxBuilder.GridSample(const Node: TOnnxNode): TGraphValue;
const
  AxisNames: array[0..3] of UTF8String = ('batch', 'channel', 'y', 'x');
var
  Source, Grid: TGraphValue;
  ImageTensor, GridTensor, Compute, X, Y, X0, Y0, X1, Y1, WX, WY, IX, IY, One, Value: TTVMFFIAny;
  Axes, Iter: array[0..3] of TTVMFFIAny;
  Shape: TArray<Int64>;
  I: Integer;
  AlignCorners: Boolean;

  function FloatImm(const Number: Single): TTVMFFIAny;
  begin
    Result := Make('ir.FloatImm',
      [TTVMRuntime.AnyDataType(FFloatDType), TTVMRuntime.AnyFloat(Number), FEmpty]);
  end;

  function Coordinate(const Component: Integer; const Extent: Int64): TTVMFFIAny;
  var
    GridValue: TTVMFFIAny;
  begin
    GridValue := ReadTensor(GridTensor, [Iter[0], Iter[2], Iter[3], IntImm(Component)]);
    if AlignCorners then
      Exit(Primitive('tirx.Mul', Primitive('tirx.Add', GridValue, One), FloatImm((Extent - 1) / 2)));
    Result := Primitive('tirx.Sub',
      Primitive('tirx.Mul', Primitive('tirx.Add', GridValue, One), FloatImm(Extent / 2)),
      FloatImm(0.5));
  end;

  function Sample(const SX, SY, WeightX, WeightY: TTVMFFIAny): TTVMFFIAny;
  var
    InsideX, InsideY, Inside, Pixel: TTVMFFIAny;
  begin
    InsideX := Primitive('tirx.And', Primitive('tirx.LE', IntImm(0), SX),
      Primitive('tirx.LT', SX, IntImm(Source.Shape[3])));
    InsideY := Primitive('tirx.And', Primitive('tirx.LE', IntImm(0), SY),
      Primitive('tirx.LT', SY, IntImm(Source.Shape[2])));
    Inside := Primitive('tirx.And', InsideX, InsideY);
    Pixel := ReadTensor(ImageTensor, [Iter[0], Iter[1], SY, SX]);
    Pixel := Intrinsic('tirx.if_then_else', FFloatType, [Inside, Pixel, FZero]);
    Result := Primitive('tirx.Mul', Pixel, Primitive('tirx.Mul', WeightX, WeightY));
  end;

begin
  Source := Input(Node, 0);
  Grid := Input(Node, 1);
  RequireCpu(
    (Source.DataType = OnnxFloat) and (Grid.DataType = OnnxFloat) and (Length(Source.Shape) = 4) and (Length(Grid.Shape) = 4) and (Grid.Shape[0] = Source.Shape[0]) and (Grid.Shape[3] = 2) and (IntAttribute(Node, 'align_corners', 0) in [0, 1]) and (StringAttribute(Node, 'mode', 'bilinear') = 'bilinear') and (StringAttribute(Node, 'padding_mode', 'zeros') = 'zeros'),
    'Unsupported ONNX GridSample signature'
  );
  AlignCorners := IntAttribute(Node, 'align_corners', 0) = 1;
  Shape := [Source.Shape[0], Source.Shape[1], Grid.Shape[1], Grid.Shape[2]];
  ImageTensor := Placeholder(Source.Shape, 'image');
  GridTensor := Placeholder(Grid.Shape, 'grid');
  for I := 0 to High(Axes) do Axes[I] := Iteration(AxisNames[I], Shape[I], False, Iter[I]);
  One := FloatImm(1);
  X := Coordinate(0, Source.Shape[3]);
  Y := Coordinate(1, Source.Shape[2]);
  X0 := Intrinsic('tirx.floor', FFloatType, [X]);
  Y0 := Intrinsic('tirx.floor', FFloatType, [Y]);
  WX := Primitive('tirx.Sub', X, X0);
  WY := Primitive('tirx.Sub', Y, Y0);
  IX := Primitive('tirx.Sub', One, WX);
  IY := Primitive('tirx.Sub', One, WY);
  X0 := Make('tirx.Cast', [FIntType, X0, FEmpty]);
  Y0 := Make('tirx.Cast', [FIntType, Y0, FEmpty]);
  X1 := Primitive('tirx.Add', X0, IntImm(1));
  Y1 := Primitive('tirx.Add', Y0, IntImm(1));
  Value := Primitive('tirx.Add',
    Primitive('tirx.Add', Sample(X0, Y0, IX, IY), Sample(X1, Y0, WX, IY)),
    Primitive('tirx.Add', Sample(X0, Y1, IX, WY), Sample(X1, Y1, WX, WY)));
  Compute := ComputeTensor('grid_sample_bilinear', Value, Axes);
  Result := Emit([Source, Grid], [ImageTensor, GridTensor], Compute, Shape);
end;

function TOnnxBuilder.ScatterND(const Node: TOnnxNode): TGraphValue;
var
  Source, Indices, Updates: TGraphValue;
  InputTensor, UpdateTensor, Value, Match, Term, Update: TTVMFFIAny;
  Axes, Iter, ReadIndices: TArray<TTVMFFIAny>;
  PrefixRank, IndexRank, SourceRank, I, J: Integer;
  Count, Position, Coordinate, Entry, BlockSize, Offset: Int64;
  Floats: TArray<Single>;
begin
  Source := Input(Node, 0);
  Indices := Input(Node, 1);
  Updates := Input(Node, 2);
  PrefixRank := Length(Indices.Shape) - 1;
  SourceRank := Length(Source.Shape);
  RequireCpu(
    (Source.DataType = Updates.DataType) and Indices.StaticInt and (Indices.DataType = OnnxInt64) and (PrefixRank >= 0) and (StringAttribute(Node, 'reduction', 'none') = 'none'),
    'ONNX ScatterND needs constant indices and replacement updates'
  );
  IndexRank := Indices.Shape[PrefixRank];
  RequireCpu((IndexRank > 0) and (IndexRank <= SourceRank) and (Length(Updates.Shape) = PrefixRank + SourceRank - IndexRank),
    'ONNX ScatterND update rank differs');
  for I := 0 to High(Updates.Shape) do
    if I < PrefixRank then
      RequireCpu(Updates.Shape[I] = Indices.Shape[I], 'ONNX ScatterND update dimensions differ')
    else
      RequireCpu(Updates.Shape[I] = Source.Shape[I - PrefixRank + IndexRank],
        'ONNX ScatterND update dimensions differ');
  Count := Length(Indices.Integers) div IndexRank;
  if Source.StaticFloat and Updates.StaticFloat then
  begin
    Floats := Copy(Source.Floats);
    BlockSize := TensorElements(Copy(Source.Shape, IndexRank, SourceRank - IndexRank));
    for Entry := 0 to Count - 1 do
    begin
      Offset := 0;
      for J := 0 to IndexRank - 1 do
      begin
        Coordinate := Indices.Integers[Entry * IndexRank + J];
        if Coordinate < 0 then Inc(Coordinate, Source.Shape[J]);
        RequireCpu((Coordinate >= 0) and (Coordinate < Source.Shape[J]),
          'ONNX ScatterND index is out of range');
        Offset := Offset * Source.Shape[J] + Coordinate;
      end;
      Offset := Offset * BlockSize;
      Move(Updates.Floats[Entry * BlockSize], Floats[Offset], BlockSize * SizeOf(Single));
    end;
    Exit(ConstantValue(Floats, Source.Shape));
  end;
  InputTensor := Placeholder(Source.Shape, 'source', Source.DataType);
  UpdateTensor := Placeholder(Updates.Shape, 'updates', Updates.DataType);
  SetLength(Iter, SourceRank);
  SetLength(Axes, SourceRank);
  for I := 0 to SourceRank - 1 do
    Axes[I] := Iteration(UTF8Encode('axis' + IntToStr(I)), Source.Shape[I], False, Iter[I]);
  Value := ReadTensor(InputTensor, Iter);
  SetLength(ReadIndices, Length(Updates.Shape));
  for I := PrefixRank to High(ReadIndices) do ReadIndices[I] := Iter[I - PrefixRank + IndexRank];
  for Entry := 0 to Count - 1 do
  begin
    Match := FEmpty;
    for J := 0 to IndexRank - 1 do
    begin
      Coordinate := Indices.Integers[Entry * IndexRank + J];
      if Coordinate < 0 then Inc(Coordinate, Source.Shape[J]);
      RequireCpu((Coordinate >= 0) and (Coordinate < Source.Shape[J]),
        'ONNX ScatterND index is out of range');
      Term := Primitive('tirx.EQ', Iter[J], IntImm(Coordinate));
      if J = 0 then
        Match := Term
      else
        Match := Primitive('tirx.And', Match, Term);
    end;
    Position := Entry;
    for I := PrefixRank - 1 downto 0 do
    begin
      ReadIndices[I] := IntImm(Position mod Indices.Shape[I]);
      Position := Position div Indices.Shape[I];
    end;
    Update := ReadTensor(UpdateTensor, ReadIndices);
    Value := Intrinsic('tirx.if_then_else', PrimitiveType(Source.DataType), [Match, Update, Value]);
  end;
  Result := Emit([Source, Updates], [InputTensor, UpdateTensor],
    ComputeTensor('scatter_nd', Value, Axes), Source.Shape, Source.DataType);
end;

function TOnnxBuilder.Transpose(const Node: TOnnxNode): TGraphValue;
var
  Source: TGraphValue;
  Shape, Axes, Values, SourceStrides: TArray<Int64>;
  Seen: TArray<Boolean>;
  InputTensor, Compute: TTVMFFIAny;
  I, J: Integer;
  Position, SourceIndex: Int64;
  OnlyMatMul, SwapsLastTwo: Boolean;
begin
  Source := Input(Node, 0);
  SetLength(Axes, Length(Source.Shape));
  for I := 0 to High(Axes) do Axes[I] := High(Axes) - I;
  Axes := IntsAttribute(Node, 'perm', Axes);
  RequireCpu(Length(Axes) = Length(Source.Shape), 'Invalid ONNX Transpose rank');
  SetLength(Shape, Length(Axes));
  SetLength(Seen, Length(Axes));
  for I := 0 to High(Axes) do
  begin
    RequireCpu((Axes[I] >= 0) and (Axes[I] < Length(Axes)), 'Invalid ONNX Transpose axis');
    RequireCpu(not Seen[Axes[I]], 'Repeated ONNX Transpose axis');
    Seen[Axes[I]] := True;
    Shape[I] := Source.Shape[Axes[I]];
  end;
  if Source.StaticInt then
  begin
    SetLength(SourceStrides, Length(Shape));
    Position := 1;
    for I := High(Shape) downto 0 do
    begin
      SourceStrides[I] := Position;
      Position := Position * Source.Shape[I];
    end;
    SetLength(Values, Length(Source.Integers));
    for I := 0 to High(Values) do
    begin
      Position := I;
      SourceIndex := 0;
      for J := High(Shape) downto 0 do
      begin
        Inc(SourceIndex, (Position mod Shape[J]) * SourceStrides[Axes[J]]);
        Position := Position div Shape[J];
      end;
      Values[I] := Source.Integers[SourceIndex];
    end;
    Exit(StaticIntegers(Shape, Values, Source.DataType));
  end;
  SwapsLastTwo := (Length(Axes) >= 2) and (Axes[High(Axes) - 1] = High(Axes)) and
    (Axes[High(Axes)] = High(Axes) - 1);
  for I := 0 to High(Axes) - 2 do SwapsLastTwo := SwapsLastTwo and (Axes[I] = I);
  if SwapsLastTwo and (Source.DataType = OnnxFloat) and
    FMatMulInputs.TryGetValue(Node.Outputs[0], OnlyMatMul) and OnlyMatMul then
    begin
      Result := Source;
      Result.Shape := Shape;
      Result.Transposed := True;
      Exit;
    end;
  InputTensor := Placeholder(Source.Shape, 'input', Source.DataType);
  Compute := Make('topi.transpose', [InputTensor, IndexArray(Axes)]);
  Result := Emit([Source], [InputTensor], Compute, Shape, Source.DataType);
end;

function TOnnxBuilder.Softmax(const Node: TOnnxNode): TGraphValue;
var
  Source, Maximum, Centered, Numerator, Total: TGraphValue;
  InputTensor, ReducedTensor, Compute, Axes: TTVMFFIAny;
  Axis, Rows, Columns: Int64;
  I: Integer;
  ReducedShape, OriginalShape: TArray<Int64>;
begin
  Source := Input(Node, 0);
  if FOpset < 13 then
    Axis := IntAttribute(Node, 'axis', 1)
  else
    Axis := IntAttribute(Node, 'axis', -1);
  if Axis < 0 then Inc(Axis, Length(Source.Shape));
  RequireCpu((Axis >= 0) and (Axis < Length(Source.Shape)), 'Invalid ONNX Softmax axis');
  OriginalShape := nil;
  if (FOpset < 13) and (Axis <> High(Source.Shape)) then
  begin
    OriginalShape := Copy(Source.Shape);
    Rows := 1;
    Columns := 1;
    for I := 0 to High(Source.Shape) do
      if I < Axis then
        Rows := Rows * Source.Shape[I]
      else
        Columns := Columns * Source.Shape[I];
    Source := EmitReshape(Source, [Rows, Columns]);
    Axis := 1;
  end;
  InputTensor := Placeholder(Source.Shape, 'input');
  if FCuda then
  begin
    ReducedShape := Copy(Source.Shape);
    ReducedShape[Axis] := 1;
    Axes := IndexArray([Axis]);
    Compute := Make('topi.max', [InputTensor, Axes, TTVMRuntime.AnyBool(True)]);
    Maximum := Emit([Source], [InputTensor], Compute, ReducedShape);
    ReducedTensor := Placeholder(ReducedShape, 'maximum');
    Compute := Make('topi.subtract', [InputTensor, ReducedTensor]);
    Centered := Emit([Source, Maximum], [InputTensor, ReducedTensor], Compute, Source.Shape);
    InputTensor := Placeholder(Source.Shape, 'centered');
    Numerator := Emit([Centered], [InputTensor], Make('topi.exp', [InputTensor]), Source.Shape);
    InputTensor := Placeholder(Source.Shape, 'exponent');
    Compute := Make('topi.sum', [InputTensor, Axes, TTVMRuntime.AnyBool(True)]);
    Total := Emit([Numerator], [InputTensor], Compute, ReducedShape);
    ReducedTensor := Placeholder(ReducedShape, 'sum');
    Compute := Make('topi.divide', [InputTensor, ReducedTensor]);
    Result := Emit([Numerator, Total], [InputTensor, ReducedTensor], Compute, Source.Shape);
    if Length(OriginalShape) > 0 then Result := EmitReshape(Result, OriginalShape);
    Exit;
  end;
  Compute := Make('topi.nn.softmax', [InputTensor, TTVMRuntime.AnyInt(Axis)]);
  Result := Emit([Source], [InputTensor], Compute, Source.Shape);
  if Length(OriginalShape) > 0 then Result := EmitReshape(Result, OriginalShape);
end;

function TOnnxBuilder.ResizeLinear(const Node: TOnnxNode): TGraphValue;
var
  Source, Scale: TGraphValue;
  Shape: TArray<Int64>;
  Axes, Iter: array[0..3] of TTVMFFIAny;
  Lower, Upper, Weight: array[0..1] of TTVMFFIAny;
  InputTensor, Coordinate, FloorValue, Ratio, One, Value, Top, Bottom: TTVMFFIAny;
  I: Integer;
  ScaledExtent: Double;

  function FloatImm(const Number: Single): TTVMFFIAny;
  begin
    Result := Make('ir.FloatImm',
      [TTVMRuntime.AnyDataType(FFloatDType), TTVMRuntime.AnyFloat(Number), FEmpty]);
  end;

  function Sample(const Y, X: TTVMFFIAny): TTVMFFIAny;
  begin
    Result := ReadTensor(InputTensor, [Iter[0], Iter[1], Y, X]);
  end;

  function Blend(const A, B, Fraction: TTVMFFIAny): TTVMFFIAny;
  begin
    Result := Primitive('tirx.Add', Primitive('tirx.Mul', A, Primitive('tirx.Sub', One, Fraction)),
      Primitive('tirx.Mul', B, Fraction));
  end;

begin
  Source := Input(Node, 0);
  RequireCpu((Source.DataType = OnnxFloat) and (Length(Source.Shape) = 4) and
    (Length(Node.Inputs) in [3, 4]) and
    (StringAttribute(Node, 'coordinate_transformation_mode', 'half_pixel') = 'align_corners') and
    (IntAttribute(Node, 'antialias', 0) = 0), 'Unsupported ONNX linear Resize signature');
  if (Length(Node.Inputs) = 4) and (Node.Inputs[3] <> '') then
    Shape := StaticInput(Node, 3)
  else begin
    Scale := Input(Node, 2);
    RequireCpu(Scale.StaticFloat and (Length(Scale.Floats) = 4),
      'ONNX linear Resize needs constant scales');
    SetLength(Shape, 4);
    for I := 0 to 3 do
    begin
      RequireCpu(not IsNan(Scale.Floats[I]) and not IsInfinite(Scale.Floats[I]) and
        (Scale.Floats[I] > 0) and (Scale.Floats[I] <= High(Integer)), 'Invalid ONNX Resize scale');
      ScaledExtent := Scale.Floats[I];
      Shape[I] := Floor(Source.Shape[I] * ScaledExtent);
    end;
  end;
  RequireCpu((Length(Shape) = 4) and (Shape[0] = Source.Shape[0]) and (Shape[1] = Source.Shape[1]),
    'ONNX Resize changes batch or channels');
  TensorElements(Shape);
  for I := 0 to 3 do
    Axes[I] := Iteration(UTF8Encode('axis' + IntToStr(I)), Shape[I], False, Iter[I]);
  One := FloatImm(1);
  InputTensor := Placeholder(Source.Shape, 'input');
  for I := 0 to 1 do
  begin
    Coordinate := FZero;
    if Shape[I + 2] > 1 then
    begin
      Ratio := FloatImm((Source.Shape[I + 2] - 1) / (Shape[I + 2] - 1));
      Coordinate := Primitive('tirx.Mul', Make('tirx.Cast', [FFloatType, Iter[I + 2], FEmpty]), Ratio);
    end;
    Coordinate := Primitive('tirx.Min', Coordinate, FloatImm(Source.Shape[I + 2] - 1));
    FloorValue := Intrinsic('tirx.floor', FFloatType, [Coordinate]);
    Weight[I] := Primitive('tirx.Sub', Coordinate, FloorValue);
    Lower[I] := Make('tirx.Cast', [FIntType, FloorValue, FEmpty]);
    Upper[I] := Primitive('tirx.Min', Primitive('tirx.Add', Lower[I], IntImm(1)),
      IntImm(Source.Shape[I + 2] - 1));
  end;
  Top := Blend(Sample(Lower[0], Lower[1]), Sample(Lower[0], Upper[1]), Weight[1]);
  Bottom := Blend(Sample(Upper[0], Lower[1]), Sample(Upper[0], Upper[1]), Weight[1]);
  Value := Blend(Top, Bottom, Weight[0]);
  Result := Emit([Source], [InputTensor], ComputeTensor('resize_linear_align_corners', Value, Axes), Shape);
end;

function TOnnxBuilder.Resize(const Node: TOnnxNode): TGraphValue;
var
  Source: TGraphValue;
  ScaleTensor: TOnnxTensor;
  Scales: TArray<Single>;
  Shape, Sizes: TArray<Int64>;
  Axes, Indices, Iter: array[0..3] of TTVMFFIAny;
  InputTensor, Compute: TTVMFFIAny;
  I: Integer;
  Numerator, Denominator: Int64;
  UseSizes: Boolean;
begin
  if StringAttribute(Node, 'mode', 'nearest') = 'linear' then Exit(ResizeLinear(Node));
  Source := Input(Node, 0);
  RequireCpu(
    (Length(Source.Shape) = 4) and (Length(Node.Inputs) in [3, 4]) and (StringAttribute(Node, 'mode', 'nearest') = 'nearest') and (StringAttribute(Node, 'coordinate_transformation_mode', 'half_pixel') = 'asymmetric') and (StringAttribute(Node, 'nearest_mode', 'round_prefer_floor') = 'floor'),
    'Unsupported ONNX Resize signature'
  );
  UseSizes := (Length(Node.Inputs) = 4) and (Node.Inputs[3] <> '');
  if UseSizes then
  begin
    Sizes := StaticInput(Node, 3);
    RequireCpu((Length(Sizes) = 4) and (Sizes[0] = Source.Shape[0]) and (Sizes[1] = Source.Shape[1]),
      'ONNX Resize changes batch or channels');
  end
  else begin
    RequireCpu(FInitializers.TryGetValue(Node.Inputs[2], ScaleTensor), 'ONNX Resize needs constant scales');
    Scales := ReadOnnxFloat32(ScaleTensor);
    RequireCpu((Length(Scales) = 4) and (Scales[0] = 1) and (Scales[1] = 1),
      'ONNX Resize changes batch or channels');
  end;
  SetLength(Shape, 4);
  for I := 0 to 3 do
  begin
    if UseSizes then
    begin
      RequireCpu(Sizes[I] > 0, 'Invalid ONNX Resize dimension');
      Shape[I] := Sizes[I];
      Numerator := Source.Shape[I];
      Denominator := Sizes[I];
    end
    else begin
      RequireCpu((Scales[I] >= 1) and (Scales[I] <= High(Integer)) and (Frac(Scales[I]) = 0),
        'ONNX Resize needs integer enlargement factors');
      Numerator := 1;
      Denominator := Trunc(Scales[I]);
      RequireCpu(Source.Shape[I] <= High(Int64) div Denominator, 'ONNX Resize dimension is too large');
      Shape[I] := Source.Shape[I] * Denominator;
    end;
    Axes[I] := Iteration(UTF8Encode('axis' + IntToStr(I)), Shape[I], False, Iter[I]);
    Indices[I] := Iter[I];
    if Numerator <> Denominator then
    begin
      if Numerator <> 1 then Indices[I] := Primitive('tirx.Mul', Indices[I], IntImm(Numerator));
      if Denominator <> 1 then
        Indices[I] := Primitive('tirx.FloorDiv', Indices[I], IntImm(Denominator));
    end;
  end;
  InputTensor := Placeholder(Source.Shape, 'input');
  Compute := ComputeTensor('resize_nearest', ReadTensor(InputTensor, Indices), Axes);
  Result := Emit([Source], [InputTensor], Compute, Shape);
end;

function TOnnxBuilder.Slice(const Node: TOnnxNode): TGraphValue;
var
  Source: TGraphValue;
  Starts, Ends, Axes, Steps, Shape, Offset, Stride, Values: TArray<Int64>;
  Iter, IterAxes, Indices: TArray<TTVMFFIAny>;
  InputTensor, Compute: TTVMFFIAny;
  Seen: TArray<Boolean>;
  I, J, Count: Integer;
  Axis, First, Last, Position, SourceIndex, SourceStride: Int64;

  function Normalize(const Value, Dimension: Int64; const Decreasing, EndIndex: Boolean): Int64;
  var
    First, Last: Int64;
  begin
    Result := Value;
    if Result < 0 then Inc(Result, Dimension);
    First := 0;
    Last := Dimension;
    if Decreasing then
    begin
      Last := Dimension - 1;
      if EndIndex then First := -1;
    end;
    if Result < First then Exit(First);
    if Result > Last then Result := Last;
  end;

begin
  Source := Input(Node, 0);
  RequireCpu(Length(Node.Inputs) in [3, 4, 5], 'Unsupported ONNX Slice signature');
  Starts := StaticInput(Node, 1);
  Ends := StaticInput(Node, 2);
  Count := Length(Starts);
  SetLength(Axes, Count);
  SetLength(Steps, Count);
  for I := 0 to Count - 1 do
  begin
    Axes[I] := I;
    Steps[I] := 1;
  end;
  if (Length(Node.Inputs) > 3) and (Node.Inputs[3] <> '') then Axes := StaticInput(Node, 3);
  if (Length(Node.Inputs) > 4) and (Node.Inputs[4] <> '') then Steps := StaticInput(Node, 4);
  RequireCpu((Length(Ends) = Count) and (Length(Axes) = Count) and (Length(Steps) = Count),
    'ONNX Slice index arrays have different lengths');
  Shape := Copy(Source.Shape);
  SetLength(Offset, Length(Shape));
  SetLength(Stride, Length(Shape));
  SetLength(Seen, Length(Shape));
  for I := 0 to High(Stride) do Stride[I] := 1;
  for I := 0 to Count - 1 do
  begin
    Axis := Axes[I];
    if Axis < 0 then Inc(Axis, Length(Shape));
    RequireCpu((Axis >= 0) and (Axis < Length(Shape)) and not Seen[Axis], 'Invalid ONNX Slice axis');
    RequireCpu(Steps[I] <> 0, 'ONNX Slice step cannot be zero');
    Seen[Axis] := True;
    First := Normalize(Starts[I], Source.Shape[Axis], Steps[I] < 0, False);
    Last := Normalize(Ends[I], Source.Shape[Axis], Steps[I] < 0, True);
    Offset[Axis] := First;
    Stride[Axis] := Steps[I];
    Shape[Axis] := 0;
    if Steps[I] > 0 then
    begin
      if Last > First then Shape[Axis] := (Last - First - 1) div Steps[I] + 1;
    end
    else if First > Last then
      begin
        if Steps[I] = Low(Int64) then
          Shape[Axis] := 1
        else
          Shape[Axis] := (First - Last - 1) div (-Steps[I]) + 1;
      end;
  end;
  if Source.StaticInt then
  begin
    Position := 1;
    for J := 0 to High(Shape) do
      if Shape[J] = 0 then Position := 0;
    if Position <> 0 then Position := TensorElements(Shape);
    SetLength(Values, Position);
    for I := 0 to High(Values) do
    begin
      Position := I;
      SourceIndex := 0;
      SourceStride := 1;
      for J := High(Shape) downto 0 do
      begin
        Inc(SourceIndex, (Offset[J] + (Position mod Shape[J]) * Stride[J]) * SourceStride);
        Position := Position div Shape[J];
        SourceStride := SourceStride * Source.Shape[J];
      end;
      Values[I] := Source.Integers[SourceIndex];
    end;
    Exit(StaticIntegers(Shape, Values, Source.DataType));
  end;
  TensorElements(Shape);
  SetLength(Iter, Length(Shape));
  SetLength(IterAxes, Length(Shape));
  SetLength(Indices, Length(Shape));
  for I := 0 to High(Shape) do
  begin
    IterAxes[I] := Iteration(UTF8Encode('axis' + IntToStr(I)), Shape[I], False, Iter[I]);
    Indices[I] := Iter[I];
    if Stride[I] <> 1 then Indices[I] := Primitive('tirx.Mul', Indices[I], IntImm(Stride[I]));
    if Offset[I] <> 0 then Indices[I] := Primitive('tirx.Add', Indices[I], IntImm(Offset[I]));
  end;
  InputTensor := Placeholder(Source.Shape, 'input', Source.DataType);
  Compute := ComputeTensor('slice', ReadTensor(InputTensor, Indices), IterAxes);
  Result := Emit([Source], [InputTensor], Compute, Shape, Source.DataType);
end;

procedure TOnnxBuilder.Split(const Node: TOnnxNode);
var
  Source, Value: TGraphValue;
  InputTensor, Outputs, Compute, Splits: TTVMFFIAny;
  Shape, Sizes, Ends: TArray<Int64>;
  Tensor: TOnnxTensor;
  Axis, Total: Int64;
  I, Count: Integer;
begin
  Count := Length(Node.Outputs);
  RequireCpu(Count > 0, 'ONNX Split has no outputs');
  Source := Input(Node, 0);
  Axis := IntAttribute(Node, 'axis', 0);
  if Axis < 0 then Inc(Axis, Length(Source.Shape));
  RequireCpu((Axis >= 0) and (Axis < Length(Source.Shape)), 'Invalid ONNX Split axis');
  SetLength(Sizes, Count);
  if Length(Node.Inputs) > 1 then
  begin
    RequireCpu(FInitializers.TryGetValue(Node.Inputs[1], Tensor), 'ONNX Split needs constant sizes');
    Sizes := ReadOnnxInt64(Tensor);
    RequireCpu(Length(Sizes) = Count, 'ONNX Split sizes do not match the outputs');
  end
  else begin
    for I := 0 to High(Sizes) do Sizes[I] := Source.Shape[Axis] div Count;
    Sizes := IntsAttribute(Node, 'split', Sizes);
  end;
  SetLength(Ends, Count - 1);
  Total := 0;
  for I := 0 to High(Sizes) do
  begin
    RequireCpu((Sizes[I] > 0) and (Sizes[I] <= Source.Shape[Axis] - Total), 'Invalid ONNX Split size');
    Inc(Total, Sizes[I]);
    if I < High(Sizes) then Ends[I] := Total;
  end;
  RequireCpu(Total = Source.Shape[Axis], 'ONNX Split sizes do not cover the input');
  InputTensor := Placeholder(Source.Shape, 'input');
  Splits := ShapeArray(Ends);
  Outputs := Make('topi.split', [InputTensor, Splits, TTVMRuntime.AnyInt(Axis)]);
  for I := 0 to Count - 1 do
  begin
    RequireCpu(Node.Outputs[I] <> '', 'ONNX Split output is missing a name');
    Compute := Make('ffi.ArrayGetItem', [Outputs, TTVMRuntime.AnyInt(I)]);
    Shape := Copy(Source.Shape);
    Shape[Axis] := Sizes[I];
    Value := Emit([Source], [InputTensor], Compute, Shape);
    FValues.Add(Node.Outputs[I], Value);
  end;
end;

procedure TOnnxBuilder.CudaTopK(const Node: TOnnxNode; const Source: TGraphValue;
  const Axis: Integer; const Count: Int64);
const
  IndexBase = Int64(4294967296);
var
  Keys, Indices: TGraphValue;
  NetworkShape, ResultShape: TArray<Int64>;
  Iter, Subscripts, Peer: TArray<TTVMFFIAny>;
  AxisIndex, SourceTensor, KeyTensor, IndexTensor, Bits, Rank, Key, OtherKey,
    PreferLow, FirstInPair, Ascending, Output: TTVMFFIAny;
  Int32: TDLDataType;
  Extent, Width, Stride: Int64;
  I: Integer;

  function EvenGroup(const Span: Int64): TTVMFFIAny;
  begin
    Result := Primitive('tirx.EQ', Primitive('tirx.FloorMod',
      Primitive('tirx.FloorDiv', AxisIndex, IntImm(Span)), IntImm(2)), IntImm(0));
  end;

begin
  Extent := Source.Shape[Axis];
  RequireCpu(Extent <= High(Integer), 'CUDA TopK axis is too large');
  Width := 1;
  while Width < Extent do Width := Width * 2;
  NetworkShape := Copy(Source.Shape);
  NetworkShape[Axis] := Width;
  ResultShape := Copy(Source.Shape);
  ResultShape[Axis] := Count;
  SetLength(Iter, Length(NetworkShape));
  SetLength(Subscripts, Length(NetworkShape));
  for I := 0 to High(Iter) do
    Iter[I] := Iteration(UTF8Encode('i' + IntToStr(I)), NetworkShape[I], False, Subscripts[I]);
  AxisIndex := Subscripts[Axis];
  SourceTensor := Placeholder(Source.Shape, 'topk_input');
  Int32 := FIntDType;
  Int32.bits := 32;
  Bits := Intrinsic('tirx.reinterpret', Make('ir.PrimType', [TTVMRuntime.AnyDataType(Int32)]),
    [ReadTensor(SourceTensor, Subscripts)]);
  Bits := Make('tirx.Cast', [FIntType, Bits, FEmpty]);
  // Order IEEE floats by signed bits, with the source index breaking equal-score ties.
  Rank := Intrinsic('tirx.if_then_else', FIntType, [Primitive('tirx.LT', Bits, IntImm(0)),
    Primitive('tirx.Sub', IntImm(Low(Integer)), Bits), Bits]);
  if IntAttribute(Node, 'largest', 1) <> 0 then Rank := Primitive('tirx.Sub', IntImm(0), Rank);
  Key := Primitive('tirx.Add', Primitive('tirx.Mul', Rank, IntImm(IndexBase)), AxisIndex);
  Key := Intrinsic('tirx.if_then_else', FIntType, [Primitive('tirx.LT', AxisIndex, IntImm(Extent)),
    Key, IntImm(High(Int64))]);
  Output := ComputeTensor('topk_keys', Key, Iter);
  Keys := Emit([Source], [SourceTensor], Output, NetworkShape, OnnxInt64);
  Peer := Copy(Subscripts);
  Width := 2;
  while Width <= NetworkShape[Axis] do
  begin
    Ascending := EvenGroup(Width);
    Stride := Width div 2;
    while Stride > 0 do
    begin
      FirstInPair := EvenGroup(Stride);
      Peer[Axis] := Intrinsic('tirx.if_then_else', FIntType, [FirstInPair,
        Primitive('tirx.Add', AxisIndex, IntImm(Stride)),
        Primitive('tirx.Sub', AxisIndex, IntImm(Stride))]);
      KeyTensor := Placeholder(NetworkShape, 'topk_pass', OnnxInt64);
      Key := ReadTensor(KeyTensor, Subscripts);
      OtherKey := ReadTensor(KeyTensor, Peer);
      PreferLow := Primitive('tirx.EQ', Ascending, FirstInPair);
      Output := ComputeTensor('topk_sort', Intrinsic('tirx.if_then_else', FIntType, [PreferLow,
        Primitive('tirx.Min', Key, OtherKey), Primitive('tirx.Max', Key, OtherKey)]), Iter);
      Keys := Emit([Keys], [KeyTensor], Output, NetworkShape, OnnxInt64);
      Stride := Stride div 2;
    end;
    Width := Width * 2;
  end;
  for I := 0 to High(Iter) do
    Iter[I] := Iteration(UTF8Encode('i' + IntToStr(I)), ResultShape[I], False, Subscripts[I]);
  KeyTensor := Placeholder(NetworkShape, 'topk_sorted_keys', OnnxInt64);
  Output := ComputeTensor('topk_indices',
    Primitive('tirx.FloorMod', ReadTensor(KeyTensor, Subscripts), IntImm(IndexBase)), Iter);
  Indices := Emit([Keys], [KeyTensor], Output, ResultShape, OnnxInt64);
  FValues.Add(Node.Outputs[1], Indices);
  IndexTensor := Placeholder(ResultShape, 'topk_indices', OnnxInt64);
  Peer := Copy(Subscripts);
  Peer[Axis] := ReadTensor(IndexTensor, Subscripts);
  Output := ComputeTensor('topk_scores', ReadTensor(SourceTensor, Peer), Iter);
  FValues.Add(Node.Outputs[0], Emit([Source, Indices], [SourceTensor, IndexTensor],
    Output, ResultShape));
end;

procedure TOnnxBuilder.TopK(const Node: TOnnxNode);
var
  Source, Value: TGraphValue;
  K, Shape: TArray<Int64>;
  Axis: Int64;
  I: Integer;
  HandleType, ShapePointerType, Int32Type, EmptyArray, InputTensor, InputShape, OutputShape,
    InputBuffer, ValuesBuffer, IndicesBuffer,
    InputPack, ValuesPack, IndicesPack, Operation, Call, Statement, Both, Scores, Indices: TTVMFFIAny;
  DType: TDLDataType;

  procedure DeclareBuffer(const Name: PAnsiChar; const Dims: TArray<Int64>;
    const DimArray, Kind, Zero: TTVMFFIAny; out Buffer, PackedTensor: TTVMFFIAny);
  var
    BufferType, Data, PackedDims: TTVMFFIAny;
  begin
    BufferType := Make(
      'tirx.BufferType',
      [
        TTVMRuntime.AnyCStr(''),
        Kind,
        DimArray,
        EmptyArray,
        FEmpty,
        TTVMRuntime.AnyInt(8),
        TTVMRuntime.AnyInt(0),
        FEmpty,
        EmptyArray,
        FEmpty
      ]
    );
    Buffer := Make('tirx.BufferVar', [TTVMRuntime.AnyCStr(Name), BufferType, FEmpty]);
    Data := Make('tirx.BufferData', [Buffer]);
    PackedDims := Make('ir.Call', [ShapePointerType, Make('ir.GetOp',
      [TTVMRuntime.AnyCStr('tirx.tvm_stack_make_shape')]), DimArray, FEmpty, EmptyArray, FEmpty]);
    PackedTensor := Intrinsic('tirx.tvm_stack_make_array', HandleType,
      [Data, PackedDims, IntImm(0), IntImm(Length(Dims)), Zero, IntImm(0)]);
  end;

  function StringImm(const Text: PAnsiChar): TTVMFFIAny;
  begin
    Result := Make('tirx.StringImm', [TTVMRuntime.AnyCStr(Text), FEmpty]);
  end;

begin
  RequireCpu((Length(Node.Outputs) = 2) and (Node.Outputs[0] <> '') and (Node.Outputs[1] <> '') and
    (IntAttribute(Node, 'sorted', 1) = 1), 'ONNX TopK requires two sorted outputs');
  Source := Input(Node, 0);
  Shape := Copy(Source.Shape);
  Axis := IntAttribute(Node, 'axis', -1);
  if Axis < 0 then Inc(Axis, Length(Shape));
  K := StaticInput(Node, 1);
  RequireCpu((Source.DataType = OnnxFloat) and (Axis >= 0) and (Axis < Length(Shape)) and
    (Length(K) = 1) and (K[0] > 0) and (K[0] <= Shape[Axis]), 'Invalid ONNX TopK input');
  if FCuda then
  begin
    CudaTopK(Node, Source, Axis, K[0]);
    Exit;
  end;
  Shape[Axis] := K[0];
  EmptyArray := Make('ffi.Array', []);
  DType := FIntDType;
  DType.code := Ord(kDLOpaqueHandle);
  DType.bits := 0;
  DType.lanes := 0;
  HandleType := Make('ir.PointerType', [Make('ir.PrimType', [TTVMRuntime.AnyDataType(DType)]),
    TTVMRuntime.AnyCStr('')]);
  ShapePointerType := Make('ir.PointerType', [FIntType, TTVMRuntime.AnyCStr('')]);
  DType := FIntDType;
  DType.bits := 32;
  Int32Type := Make('ir.PrimType', [TTVMRuntime.AnyDataType(DType)]);
  InputShape := ShapeArray(Source.Shape);
  OutputShape := ShapeArray(Shape);
  InputTensor := Make('te.Placeholder', [InputShape, FFloatType, TTVMRuntime.AnyCStr('input')]);
  DeclareBuffer('topk_input', Source.Shape, InputShape, FFloatType, FZero, InputBuffer, InputPack);
  DeclareBuffer('topk_values', Shape, OutputShape, FFloatType, FZero, ValuesBuffer, ValuesPack);
  DeclareBuffer('topk_indices', Shape, OutputShape, FIntType, IntImm(0), IndicesBuffer, IndicesPack);
  Call := Intrinsic('tirx.tvm_call_packed', Int32Type, [StringImm('tvm.contrib.sort.topk'),
    InputPack, ValuesPack, IndicesPack, IntImm(K[0]), IntImm(Axis), StringImm('both'),
    Make('ir.IntImm', [TTVMRuntime.AnyDataType(DataType(OnnxBool)),
      TTVMRuntime.AnyInt(Ord(IntAttribute(Node, 'largest', 1) = 0)), FEmpty])]);
  Statement := Make('tirx.Evaluate', [Call, FEmpty]);
  Operation := Make('te.ExternOp', [TTVMRuntime.AnyCStr('topk'), TTVMRuntime.AnyCStr(''), FEmpty,
    Make('ffi.Array', [InputTensor]), Make('ffi.Array', [InputBuffer]),
    Make('ffi.Array', [ValuesBuffer, IndicesBuffer]), Statement]);
  Scores := Make('te.OpGetOutput', [Operation, TTVMRuntime.AnyInt(0)]);
  Indices := Make('te.OpGetOutput', [Operation, TTVMRuntime.AnyInt(1)]);
  Both := EmitKernel([InputTensor, Scores, Indices], [ExpressionOf(Source)],
    [TensorType(Shape), TensorType(Shape, OnnxInt64)]);
  for I := 0 to 1 do
  begin
    Value := Default(TGraphValue);
    Value.Shape := Shape;
    Value.DataType := OnnxFloat;
    if I = 1 then Value.DataType := OnnxInt64;
    Value.Expression := Make('relax.TupleGetItem', [Both, TTVMRuntime.AnyInt(I), FEmpty]);
    FValues.Add(Node.Outputs[I], Value);
  end;
end;

procedure TOnnxBuilder.AddConstant(const Node: TOnnxNode);
var
  Attribute: TOnnxAttribute;
begin
  for Attribute in Node.Attributes do
    if (Attribute.Name = 'value') and (Attribute.Tensor.DataType <> 0) then
    begin
      FInitializers.Add(Node.Outputs[0], Attribute.Tensor);
      Exit;
    end;
  raise ECpuGraph.Create('ONNX Constant needs an embedded tensor');
end;

procedure TOnnxBuilder.AddNode(const Node: TOnnxNode);

  function Lower: TGraphValue;
  begin
    if Node.OpType = 'Conv' then Exit(Conv(Node));
    if Node.OpType = 'ConvTranspose' then Exit(ConvTranspose(Node));
    if (Node.OpType = 'AveragePool') or (Node.OpType = 'MaxPool') or
      (Node.OpType = 'GlobalAveragePool') then Exit(Pool(Node));
    if Node.OpType = 'Concat' then Exit(Concat(Node));
    if Node.OpType = 'BatchNormalization' then Exit(BatchNormalization(Node));
    if Node.OpType = 'Add' then Exit(Binary(Node, 'topi.add'));
    if Node.OpType = 'Mul' then Exit(Binary(Node, 'topi.multiply'));
    if Node.OpType = 'Sub' then Exit(Binary(Node, 'topi.subtract'));
    if Node.OpType = 'Div' then Exit(Binary(Node, 'topi.divide'));
    if Node.OpType = 'Pow' then Exit(Binary(Node, 'topi.power'));
    if Node.OpType = 'Less' then Exit(Binary(Node, 'topi.less'));
    if Node.OpType = 'Greater' then Exit(Binary(Node, 'topi.greater'));
    if Node.OpType = 'GreaterOrEqual' then Exit(Binary(Node, 'topi.greater_equal'));
    if Node.OpType = 'Equal' then Exit(Binary(Node, 'topi.equal'));
    if Node.OpType = 'And' then Exit(Binary(Node, 'topi.logical_and'));
    if Node.OpType = 'Max' then
    begin
      RequireCpu(Length(Node.Inputs) = 2, 'ONNX Max requires two inputs on this path');
      Exit(Binary(Node, 'topi.maximum'));
    end;
    if Node.OpType = 'Gemm' then Exit(Gemm(Node));
    if Node.OpType = 'MatMul' then Exit(MatMul(Node));
    if Node.OpType = 'Reshape' then Exit(Reshape(Node));
    if Node.OpType = 'Shape' then Exit(ShapeOf(Node));
    if Node.OpType = 'Range' then Exit(Range(Node));
    if Node.OpType = 'Mod' then Exit(Binary(Node, ''));
    if Node.OpType = 'Cast' then Exit(Cast(Node));
    if (Node.OpType = 'Expand') or (Node.OpType = 'Tile') then Exit(ExpandOrTile(Node));
    if Node.OpType = 'ConstantOfShape' then Exit(ConstantOfShape(Node));
    if Node.OpType = 'LayerNormalization' then Exit(LayerNormalization(Node));
    if Node.OpType = 'Clip' then Exit(Clip(Node));
    if Node.OpType = 'Where' then Exit(Where(Node));
    if Node.OpType = 'GridSample' then Exit(GridSample(Node));
    if Node.OpType = 'ScatterND' then Exit(ScatterND(Node));
    if Node.OpType = 'Unsqueeze' then Exit(Unsqueeze(Node));
    if Node.OpType = 'Squeeze' then Exit(Squeeze(Node));
    if Node.OpType = 'Gather' then Exit(Gather(Node, False));
    if Node.OpType = 'GatherElements' then Exit(Gather(Node, True));
    if (Node.OpType = 'ReduceMean') or (Node.OpType = 'ReduceSum') or
      (Node.OpType = 'ReduceMax') then Exit(Reduce(Node));
    if Node.OpType = 'Pad' then Exit(Pad(Node));
    if Node.OpType = 'Flatten' then Exit(Flatten(Node));
    if Node.OpType = 'Transpose' then Exit(Transpose(Node));
    if Node.OpType = 'Softmax' then Exit(Softmax(Node));
    if Node.OpType = 'Resize' then Exit(Resize(Node));
    if Node.OpType = 'Slice' then Exit(Slice(Node));
    if Node.OpType = 'Relu' then Exit(Unary(Node, 'topi.nn.relu'));
    if Node.OpType = 'Sigmoid' then Exit(Unary(Node, 'topi.sigmoid'));
    if Node.OpType = 'Sqrt' then Exit(Unary(Node, 'topi.sqrt'));
    if Node.OpType = 'Ceil' then Exit(Unary(Node, 'topi.ceil'));
    if Node.OpType = 'Floor' then Exit(Unary(Node, 'topi.floor'));
    if Node.OpType = 'Tanh' then Exit(Unary(Node, 'topi.tanh'));
    if Node.OpType = 'Erf' then Exit(Unary(Node, 'topi.erf'));
    if Node.OpType = 'Log' then Exit(Unary(Node, 'topi.log'));
    if Node.OpType = 'Sin' then Exit(Unary(Node, 'topi.sin'));
    if Node.OpType = 'Cos' then Exit(Unary(Node, 'topi.cos'));
    if Node.OpType = 'Neg' then Exit(Unary(Node, 'topi.negative'));
    if Node.OpType = 'Not' then Exit(Unary(Node, 'topi.logical_not'));
    if Node.OpType = 'Identity' then Exit(Input(Node, 0));
    if (Node.OpType = 'Dropout') and (Length(Node.Inputs) = 1) then Exit(Input(Node, 0));
    raise ECpuGraph.Create('Unsupported ONNX operator: ' + Node.OpType);
  end;

begin
  if (Node.Domain = 'crosspascal') and (Node.OpType = 'Q8MatMul') then
  begin
    RequireCpu((Length(Node.Outputs) = 1) and (Node.Outputs[0] <> ''),
      'Invalid int8 matrix multiplication output');
    FValues.Add(Node.Outputs[0], Q8MatMul(Node));
    Exit;
  end;
  RequireCpu((Node.Domain = '') or (Node.Domain = 'ai.onnx'),
    'Unsupported ONNX operator domain: ' + Node.Domain);
  try
    if Node.OpType = 'Split' then
    begin
      Split(Node);
      Exit;
    end;
    if Node.OpType = 'TopK' then
    begin
      TopK(Node);
      Exit;
    end;
    RequireCpu((Length(Node.Outputs) = 1) and (Node.Outputs[0] <> ''), 'Unsupported ONNX output count');
    if Node.OpType = 'Constant' then
      AddConstant(Node)
    else
      FValues.Add(Node.Outputs[0], Lower);
  except
    on E: Exception do raise ECpuGraph.Create(Node.OpType + ' ' + Node.Name + ': ' + E.Message);
  end;
end;

function TOnnxBuilder.Build(const Model: TOnnxModel): TTVMFFIObjectHandle;
var
  Initializer: TOnnxTensor;
  InputInfo, OutputInfo: TOnnxValue;
  InputValue, OutputValue: TGraphValue;
  Node: TOnnxNode;
  Inputs, Outputs, OutputTypes: TArray<TTVMFFIAny>;
  Params, Block, Value, OutputType, Body, Fn: TTVMFFIAny;
  Name: UTF8String;
  I, Count, OutputIndex: Integer;
  Dim, Earlier: Int64;
  UsedOnlyByMatMul: Boolean;
begin
  RequireCpu((Model.Opset >= 7) and (Model.Opset <= 18),
    'This ONNX compiler path supports opsets 7 through 18');
  FOpset := Model.Opset;
  for Node in Model.Nodes do
    for I := 0 to High(Node.Inputs) do
      if (Node.OpType <> 'MatMul') or (I <> 1) then
        FMatMulInputs.AddOrSetValue(Node.Inputs[I], False)
      else if not FMatMulInputs.TryGetValue(Node.Inputs[I], UsedOnlyByMatMul) then
          FMatMulInputs.Add(Node.Inputs[I], True);
  for OutputInfo in Model.Outputs do FMatMulInputs.AddOrSetValue(OutputInfo.Name, False);
  for Initializer in Model.Initializers do FInitializers.Add(Initializer.Name, Initializer);
  SetLength(Inputs, Length(Model.Inputs));
  Count := 0;
  for InputInfo in Model.Inputs do
  begin
    if FInitializers.ContainsKey(InputInfo.Name) then Continue;
    RequireCpu(InputInfo.HasShape, 'Expected a shaped ONNX input');
    InputValue := Default(TGraphValue);
    InputValue.DataType := InputInfo.DataType;
    SetLength(InputValue.Shape, Length(InputInfo.Shape));
    for I := 0 to High(InputInfo.Shape) do
    begin
      Dim := InputInfo.Shape[I].Value;
      if (Dim < 0) and (I = 0) then Dim := FBatchSize;
      if InputInfo.Shape[I].Symbol <> '' then
      begin
        if FDimensions.TryGetValue(InputInfo.Shape[I].Symbol, Earlier) then
          RequireCpu(Earlier = Dim, 'ONNX input dimensions disagree: ' + InputInfo.Shape[I].Symbol)
        else
          FDimensions.Add(InputInfo.Shape[I].Symbol, Dim);
      end;
      InputValue.Shape[I] := Dim;
    end;
    TensorElements(InputValue.Shape);
    Name := UTF8Encode(InputInfo.Name);
    InputValue.Expression := Make('ir.Var',
      [TTVMRuntime.AnyCStr(PAnsiChar(Name)), TensorType(InputValue.Shape, InputValue.DataType), FEmpty]);
    FValues.Add(InputInfo.Name, InputValue);
    Inputs[Count] := InputValue.Expression;
    Inc(Count);
  end;
  RequireCpu(Count > 0, 'ONNX graph has no runtime inputs');
  SetLength(Inputs, Count);
  Params := Make('ffi.Array', Inputs);
  Invoke('relax.BlockBuilderBeginScope', [FBuilder, Params]);
  Invoke('relax.BlockBuilderBeginDataflowBlock', [FBuilder]);
  for Node in Model.Nodes do AddNode(Node);
  SetLength(Outputs, Length(Model.Outputs));
  RequireCpu(Length(Outputs) > 0, 'ONNX graph has no outputs');
  SetLength(OutputTypes, Length(Outputs));
  for OutputIndex := 0 to High(Outputs) do
  begin
    OutputInfo := Model.Outputs[OutputIndex];
    RequireCpu(FValues.TryGetValue(OutputInfo.Name, OutputValue),
      'Missing ONNX graph output: ' + OutputInfo.Name);
    RequireCpu(OutputInfo.HasShape and (OutputInfo.DataType = OutputValue.DataType) and
      (Length(OutputValue.Shape) = Length(OutputInfo.Shape)), 'ONNX graph output type differs');
    for I := 0 to High(OutputValue.Shape) do
    begin
      Dim := OutputInfo.Shape[I].Value;
      if (Dim < 0) and (OutputInfo.Shape[I].Symbol <> '') then
        if not FDimensions.TryGetValue(OutputInfo.Shape[I].Symbol, Dim) then Dim := -1;
      RequireCpu((Dim < 0) or (OutputValue.Shape[I] = Dim), 'ONNX graph output dimensions differ');
    end;
    Outputs[OutputIndex] := Make('relax.BlockBuilderEmitOutput',
      [FBuilder, ExpressionOf(OutputValue), TTVMRuntime.AnyCStr('output')]);
    OutputTypes[OutputIndex] := TensorType(OutputValue.Shape, OutputValue.DataType);
  end;
  Block := Make('relax.BlockBuilderEndBlock', [FBuilder]);
  Invoke('relax.BlockBuilderEndScope', [FBuilder]);
  Value := Outputs[0];
  OutputType := OutputTypes[0];
  if Length(Outputs) > 1 then
  begin
    Value := Make('relax.Tuple', [Make('ffi.Array', Outputs), FEmpty]);
    OutputType := Make('ir.TupleType', [Make('ffi.Array', OutputTypes), FEmpty]);
  end;
  Body := Make('relax.SeqExpr', [Make('ffi.Array', [Block]), Value, FEmpty]);
  Fn := Make('relax.Function', [Params, Body, OutputType, TTVMRuntime.AnyBool(True), FAttributes, FEmpty]);
  Make('relax.BlockBuilderAddFunction', [FBuilder, Fn, TTVMRuntime.AnyCStr('main')]);
  Result := CompilerObject('relax.BlockBuilderFinalize', [FBuilder]);
end;

procedure CompileCpuOnnx(const CompilerDirectory, InputFile, TargetJSON, KernelObject, VMObject: string;
  const BatchSize: Integer);
var
  Model: TOnnxModel;
begin
  Model := TOnnxModel.Create(InputFile);
  try
    CompileCpuOnnx(Model, CompilerDirectory, TargetJSON, KernelObject, VMObject, BatchSize);
  finally
    Model.Free;
  end;
end;

procedure CompileCpuOnnx(const Model: TOnnxModel;
  const CompilerDirectory, TargetJSON, KernelObject, VMObject: string; const BatchSize: Integer);
var
  Builder: TOnnxBuilder;
  IR: TTVMFFIObjectHandle;
begin
  InitializeTvmCompiler(CompilerDirectory);
  Builder := nil;
  IR := nil;
  try
    Builder := TOnnxBuilder.Create(BatchSize);
    IR := Builder.Build(Model);
    CompileCpuRelaxModule(IR, TargetJSON, KernelObject, VMObject);
  finally
    TTVMRuntime.ObjectRelease(IR);
    Builder.Free;
  end;
end;

procedure CompileCudaOnnx(const Model: TOnnxModel;
  const CompilerDirectory, NvrtcLibrary, Architecture, HostTargetJSON, KernelObject, VMObject: string;
  const BatchSize: Integer);
var
  Builder: TOnnxBuilder;
  IR: TTVMFFIObjectHandle;
begin
  InitializeTvmCompiler(CompilerDirectory);
  Builder := TOnnxBuilder.Create(BatchSize, True);
  IR := nil;
  try
    IR := Builder.Build(Model);
    CompileCudaRelaxModule(IR, CompilerDirectory, NvrtcLibrary, Architecture, HostTargetJSON,
      KernelObject, VMObject);
  finally
    TTVMRuntime.ObjectRelease(IR);
    Builder.Free;
  end;
end;

procedure CompileWindowsTvmOnnxLibrary(const Model: TOnnxModel; const Kit: TWindowsTvmKit;
  const LibraryFile, NvrtcLibrary, Architecture: string);
const
  WindowsLLVM = '{"kind":"llvm","mtriple":"x86_64-pc-windows-msvc","mcpu":"haswell"}';
var
  Kernel, VM, Extra: string;
  Temporary: TArray<string>;
begin
  Kernel := ChangeFileExt(LibraryFile, '.kernels.obj');
  VM := ChangeFileExt(LibraryFile, '.vm.obj');
  Temporary := [Kernel, VM, ChangeFileExt(LibraryFile, '.lib'), ChangeFileExt(LibraryFile, '.exp')];
  try
    if Architecture = '' then
      CompileCpuOnnx(Model, Kit.CompilerDirectory, WindowsLLVM, Kernel, VM)
    else
      CompileCudaOnnx(Model, Kit.CompilerDirectory, NvrtcLibrary, Architecture, WindowsLLVM, Kernel, VM);
    LinkWindowsTvmLibrary(Kit.Linker, LibraryFile, [Kernel, VM], [Kit.FfiLibrary, 'libcmt.lib'],
      [Kit.MsvcLibraries, Kit.UcrtLibraries, Kit.WindowsLibraries]);
  finally
    for Extra in Temporary do
      if FileExists(Extra) then DeleteFileW(PWideChar(Extra));
  end;
end;

end.
