{ ************************************************************************** }
{                                                                            }
{ CrossPascal.RowQuantization                                                }
{                                                                            }
{ Copyright © 2026 Yuriy Pisarev                                             }
{                                                                            }
{ ************************************************************************** }

unit CrossPascal.RowQuantization;

{$IFDEF FPC}{$MODE DELPHIUNICODE}{$ENDIF}
{$POINTERMATH ON}

interface

uses
  {$IFDEF FPC}Classes{$ELSE}System.Classes{$ENDIF};

procedure WriteSymmetricInt8Rows(const Input: PSingle; const Rows, Width: Integer;
  const Quantized, Scales: TStream);

implementation

uses
  {$IFDEF FPC}Math{$ELSE}System.Math{$ENDIF}, TVM.CpuGraph;

procedure WriteSymmetricInt8Rows(const Input: PSingle; const Rows, Width: Integer;
  const Quantized, Scales: TStream);
const
  ChunkRows = 128;
  SmallestNormal: Single = 1.1754943508222875e-38;
var
  Chunk: TArray<ShortInt>;
  ChunkScales: array[0..ChunkRows - 1] of Single;
  OldRounding: TFPURoundingMode;
  Values: PSingle;
  Offset, I, J, Count, Rounded: Integer;
  Largest, Magnitude, Scale, Ratio: Single;
begin
  RequireCpu(
    (Input <> nil) and (Rows > 0) and (Width > 0) and (Width <= High(Integer) div ChunkRows) and (Int64(Rows) * Width <= High(Int64) div SizeOf(Single)) and (Quantized <> nil) and (Scales <> nil),
    'Invalid int8 weight dimensions or output'
  );
  SetLength(Chunk, Min(Rows, ChunkRows) * Width);
  OldRounding := SetRoundMode(rmNearest);
  try
    Offset := 0;
    while Offset < Rows do
    begin
      Count := Min(Rows - Offset, ChunkRows);
      for I := 0 to Count - 1 do
      begin
        Values := Input + (Int64(Offset) + I) * Width;
        Largest := 0;
        for J := 0 to Width - 1 do
        begin
          Magnitude := Abs(Values[J]);
          RequireCpu(not IsNan(Magnitude) and not IsInfinite(Magnitude), 'Non-finite FP32 weight');
          if Magnitude > Largest then Largest := Magnitude;
        end;
        Scale := Largest / 127;
        if Scale < SmallestNormal then Scale := SmallestNormal;
        ChunkScales[I] := Scale;
        for J := 0 to Width - 1 do
        begin
          Ratio := Values[J] / Scale;
          Rounded := Round(Ratio);
          Chunk[I * Width + J] := EnsureRange(Rounded, -127, 127);
        end;
      end;
      Quantized.WriteBuffer(Chunk[0], Count * Width);
      Scales.WriteBuffer(ChunkScales[0], Count * SizeOf(Single));
      Inc(Offset, Count);
    end;
  finally
    SetRoundMode(OldRounding);
  end;
end;

end.
