{ ************************************************************************** }
{                                                                            }
{ CrossPascal.YoloSource                                                     }
{                                                                            }
{ Copyright © 2026 Yuriy Pisarev                                             }
{                                                                            }
{ ************************************************************************** }

unit CrossPascal.YoloSource;

{$IFDEF FPC}{$MODE DELPHIUNICODE}{$ENDIF}

interface

uses
  CrossPascal.Onnx;

function BindYoloCheckpoint(const Recipe: TOnnxModel; const CheckpointFile: string): TOnnxModel;

implementation

uses
  {$IFDEF FPC}SysUtils, Math{$ELSE}System.SysUtils, System.Math{$ENDIF},
  CrossPascal.TorchCheckpoint, TVM.CpuGraph;

function ReadWeight(const Source: TTorchCheckpoint; const Name: string;
  const ExpectedShape: array of Int64): TArray<Single>;
var
  Shape: TArray<Int64>;
  I: Integer;
begin
  Result := Source.ReadFloats(Name, Shape);
  RequireCpu(Length(Shape) = Length(ExpectedShape), 'Wrong PyTorch tensor rank: ' + Name);
  for I := 0 to High(Shape) do
    RequireCpu(Shape[I] = ExpectedShape[I], 'Wrong PyTorch tensor dimensions: ' + Name);
end;

function IsSuffix(const Name, Suffix: string): Boolean;
begin
  Result := Copy(Name, Length(Name) - Length(Suffix) + 1, MaxInt) = Suffix;
end;

function FoldBatchNorm(const Source: TTorchCheckpoint; const Tensor: TOnnxTensor;
  const BatchNorm: string; const IsBias: Boolean): TArray<Single>;
var
  Gamma, Variance, Mean, Beta, ConvBias: TArray<Single>;
  Scale, Sigma, Epsilon: Single;
  Channels, RowWidth, Channel, I, At: Integer;
begin
  RequireCpu((Length(Tensor.Shape) > 0) and (Tensor.Shape[0] > 0),
    'Invalid batch-norm convolution shape: ' + Tensor.Name);
  Channels := Tensor.Shape[0];
  Gamma := ReadWeight(Source, BatchNorm + '.weight', [Channels]);
  Variance := ReadWeight(Source, BatchNorm + '.running_var', [Channels]);
  Epsilon := Source.BatchNormEpsilon(BatchNorm);
  if IsBias then
  begin
    Mean := ReadWeight(Source, BatchNorm + '.running_mean', [Channels]);
    Beta := ReadWeight(Source, BatchNorm + '.bias', [Channels]);
    if Source.Contains(Tensor.Name) then
      ConvBias := ReadWeight(Source, Tensor.Name, [Channels])
    else
      SetLength(ConvBias, Channels);
    SetLength(Result, Channels);
    for Channel := 0 to Channels - 1 do
    begin
      Sigma := Sqrt(Variance[Channel] + Epsilon);
      Scale := Gamma[Channel] / Sigma;
      Result[Channel] := Scale * ConvBias[Channel] +
        (Beta[Channel] - Gamma[Channel] * Mean[Channel] / Sigma);
    end;
  end
  else begin
    Result := ReadWeight(Source, Tensor.Name, Tensor.Shape);
    RowWidth := Length(Result) div Channels;
    At := 0;
    for Channel := 0 to Channels - 1 do
    begin
      Sigma := Sqrt(Variance[Channel] + Epsilon);
      Scale := Gamma[Channel] / Sigma;
      for I := 1 to RowWidth do
      begin
        Result[At] := Result[At] * Scale;
        Inc(At);
      end;
    end;
  end;
end;

function TryFoldParallelConvolution(const Source: TTorchCheckpoint; const Tensor: TOnnxTensor;
  const IsBias: Boolean; out Values: TArray<Single>): Boolean;
var
  Base, Suffix: string;
  First, Second: TOnnxTensor;
  Other: TArray<Single>;
  Shape: TArray<Int64>;
  I, Row, Y, X, Height, Width, OtherHeight, OtherWidth, PadY, PadX: Integer;
begin
  if IsBias then
    Suffix := '.conv.bias'
  else
    Suffix := '.conv.weight';
  Base := Copy(Tensor.Name, 1, Length(Tensor.Name) - Length(Suffix));
  Result := IsSuffix(Tensor.Name, Suffix) and
    Source.Contains(Base + '.conv.conv.weight') and
    Source.Contains(Base + '.conv1.conv.weight');
  if not Result then Exit;
  First := Tensor;
  Second := Tensor;
  First.Name := Base + '.conv' + Suffix;
  Second.Name := Base + '.conv1' + Suffix;
  if not IsBias then
  begin
    Other := Source.ReadFloats(Second.Name, Shape);
    RequireCpu((Length(Tensor.Shape) = 4) and (Length(Shape) = 4),
      'Invalid parallel convolution rank: ' + Tensor.Name);
    RequireCpu((Shape[0] = Tensor.Shape[0]) and (Shape[1] = Tensor.Shape[1]) and
      (Shape[2] <= Tensor.Shape[2]) and (Shape[3] <= Tensor.Shape[3]) and
      ((Tensor.Shape[2] - Shape[2]) mod 2 = 0) and
      ((Tensor.Shape[3] - Shape[3]) mod 2 = 0),
      'Incompatible parallel convolution kernels: ' + Tensor.Name);
    Second.Shape := Shape;
  end;
  Values := FoldBatchNorm(Source, First, Base + '.conv.bn', IsBias);
  Other := FoldBatchNorm(Source, Second, Base + '.conv1.bn', IsBias);
  if IsBias then
  begin
    RequireCpu(Length(Values) = Length(Other), 'Parallel convolution channel count differs');
    for I := 0 to High(Values) do Values[I] := Values[I] + Other[I];
  end
  else begin
    Height := Tensor.Shape[2];
    Width := Tensor.Shape[3];
    OtherHeight := Shape[2];
    OtherWidth := Shape[3];
    PadY := (Height - OtherHeight) div 2;
    PadX := (Width - OtherWidth) div 2;
    I := 0;
    for Row := 0 to Tensor.Shape[0] * Tensor.Shape[1] - 1 do
      for Y := 0 to OtherHeight - 1 do
        for X := 0 to OtherWidth - 1 do
        begin
          Values[(Row * Height + Y + PadY) * Width + X + PadX] := Values[(Row * Height + Y + PadY) * Width + X + PadX] + Other[I];
          Inc(I);
        end;
  end;
end;

function BindYoloCheckpoint(const Recipe: TOnnxModel; const CheckpointFile: string): TOnnxModel;
const
  ConvWeight = '.conv.weight';
  ConvBias = '.conv.bias';
  PointwiseWeight = '.pw.weight';
  PointwiseBias = '.pw.bias';
var
  Source: TTorchCheckpoint;
  Initializers: TArray<TOnnxTensor>;
  Buffers: TArray<TBytes>;
  Values: TArray<Single>;
  Name, BatchNorm: string;
  I, J, Replaced: Integer;
  IsBias: Boolean;
begin
  Source := TTorchCheckpoint.Create(CheckpointFile);
  try
    Initializers := Copy(Recipe.Initializers);
    SetLength(Buffers, Length(Initializers));
    Replaced := 0;
    for I := 0 to High(Initializers) do
    begin
      Name := Initializers[I].Name;
      if Copy(Name, 1, 6) <> 'model.' then Continue;
      RequireCpu((Initializers[I].DataType = 1) and
        (TensorElements(Initializers[I].Shape) > 0) and (Initializers[I].RawData.Size = 0) and
        not Initializers[I].IsExternal, 'Invalid YOLO weight placeholder: ' + Name);
      IsBias := IsSuffix(Name, ConvBias) or IsSuffix(Name, PointwiseBias);
      BatchNorm := '';
      if IsSuffix(Name, ConvBias) then
        BatchNorm := Copy(Name, 1, Length(Name) - Length(ConvBias)) + '.bn';
      if IsSuffix(Name, ConvWeight) then
        BatchNorm := Copy(Name, 1, Length(Name) - Length(ConvWeight)) + '.bn';
      if IsSuffix(Name, PointwiseBias) then
        BatchNorm := Copy(Name, 1, Length(Name) - Length(PointwiseBias)) + '.bn';
      if IsSuffix(Name, PointwiseWeight) then
        BatchNorm := Copy(Name, 1, Length(Name) - Length(PointwiseWeight)) + '.bn';
      if (BatchNorm <> '') and Source.Contains(BatchNorm + '.running_var') then
        Values := FoldBatchNorm(Source, Initializers[I], BatchNorm, IsBias)
      else if Source.Contains(Name) then
        Values := ReadWeight(Source, Name, Initializers[I].Shape)
      else begin
        { The checkpoint may store both training branches of a fused convolution. }
        if not TryFoldParallelConvolution(Source, Initializers[I], IsBias, Values) then
          Values := ReadWeight(Source, Name, Initializers[I].Shape);
      end;
      for J := 0 to High(Values) do
        RequireCpu(not IsNan(Values[J]) and not IsInfinite(Values[J]), 'Nonfinite YOLO weight: ' + Name);
      SetLength(Buffers[I], Length(Values) * SizeOf(Single));
      Move(Values[0], Buffers[I][0], Length(Buffers[I]));
      Initializers[I].Serialized := Default(TOnnxBytes);
      Initializers[I].RawData.Data := @Buffers[I][0];
      Initializers[I].RawData.Size := Length(Buffers[I]);
      Inc(Replaced);
    end;
    RequireCpu(Replaced > 0, 'YOLO graph recipe does not contain weight placeholders');
    Result := TOnnxModel.CreateGenerated(Recipe.Opset, Recipe.Inputs, Recipe.Outputs, Initializers,
      Recipe.Nodes, Buffers);
  finally
    Source.Free;
  end;
end;

end.
