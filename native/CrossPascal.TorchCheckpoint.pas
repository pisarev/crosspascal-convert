{ ************************************************************************** }
{                                                                            }
{ CrossPascal.TorchCheckpoint                                                }
{                                                                            }
{ Copyright © 2026 Yuriy Pisarev                                             }
{                                                                            }
{ ************************************************************************** }

unit CrossPascal.TorchCheckpoint;

{$IFDEF FPC}{$MODE DELPHIUNICODE}{$ENDIF}
{$POINTERMATH ON}

interface

uses
  {$IFDEF FPC}Classes, SysUtils, Generics.Collections
  {$ELSE}System.Classes, System.SysUtils, System.Generics.Collections{$ENDIF};

type
  TTorchCheckpoint = class
  private type
    TStorage = record
      DataType: string;
      Key: string;
      Count: Int64;
    end;
    TTensor = record
      Storage: TStorage;
      Offset: Int64;
      Shape: TArray<Int64>;
      Strides: TArray<Int64>;
    end;
    TPickleValue = class
      Kind: (pvNone, pvText, pvNumber, pvFloat, pvGlobal, pvMap, pvSequence, pvObject, pvStorage, pvTensor);
      Text: string;
      Number: Int64;
      Scalar: Double;
      Items: TArray<TPickleValue>;
      Map: TDictionary<string, TPickleValue>;
      State: TPickleValue;
      Storage: TStorage;
      Tensor: TTensor;
      destructor Destroy; override;
    end;
  private
    FArchive: TObjectDictionary<string, TMemoryStream>;
    FPrefix: string;
    FObjects: TObjectList<TPickleValue>;
    FStack: TList<TPickleValue>;
    FMemo: TDictionary<Cardinal, TPickleValue>;
    FTensors: TDictionary<string, TTensor>;
    FEpsilon: TDictionary<string, Single>;
    FPickle: TStream;
    procedure ReadArchive(const FileName: string);
    function NewValue: TPickleValue;
    function Pop: TPickleValue;
    function Top: TPickleValue;
    function PopItems(const Count: Integer): TArray<TPickleValue>;
    function PopMarked: TArray<TPickleValue>;
    function ReadByte: Byte;
    function ReadCardinal: Cardinal;
    function ReadLine: string;
    function ReadText(const Count: Integer): string;
    function Rebuild(const Callable, Arguments: TPickleValue): TPickleValue;
    function ReadStorage(const Identity: TPickleValue): TPickleValue;
    function ReadPickle: TPickleValue;
    function Field(const Map: TPickleValue; const Name: string): TPickleValue;
    procedure ReadModule(const Model: TPickleValue; const Name: string; const Depth: Integer);
  public
    constructor Create(const FileName: string);
    destructor Destroy; override;
    function Contains(const Name: string): Boolean;
    function ReadFloats(const Name: string; out Shape: TArray<Int64>): TArray<Single>;
    function BatchNormEpsilon(const Name: string): Single;
    function TensorNames: TArray<string>;
  end;

implementation

uses
  CrossVision.Float16, TVM.CpuGraph;

const
  MaximumArchive = 256 * 1024 * 1024;

destructor TTorchCheckpoint.TPickleValue.Destroy;
begin
  Map.Free;
  inherited;
end;

procedure TTorchCheckpoint.ReadArchive(const FileName: string);
var
  Mem: TMemoryStream;
  FileStream: TFileStream;
  EndRecord: array[0..21] of Byte;
  Central: array[0..45] of Byte;
  Local: array[0..29] of Byte;
  CentralStart, CentralEnd, NextEntry, MemberOffset, MemberSize, TotalBytes: Int64;
  I, Entries, NameSize: Integer;
  Name, LocalName: UTF8String;
  MemberName: string;
begin
  FileStream := TFileStream.Create(FileName, fmOpenRead or fmShareDenyWrite);
  try
    RequireCpu((FileStream.Size >= SizeOf(EndRecord)) and (FileStream.Size <= MaximumArchive),
      'Unsupported PyTorch archive size');
    FileStream.Position := FileStream.Size - SizeOf(EndRecord);
    FileStream.ReadBuffer(EndRecord, SizeOf(EndRecord));
    RequireCpu((PCardinal(@EndRecord[0])^ = $06054B50) and (PCardinal(@EndRecord[4])^ = 0) and
      (PWord(@EndRecord[8])^ = PWord(@EndRecord[10])^) and (PWord(@EndRecord[20])^ = 0),
      'Expected a single-volume PyTorch ZIP archive');
    CentralStart := PCardinal(@EndRecord[16])^;
    CentralEnd := CentralStart + PCardinal(@EndRecord[12])^;
    Entries := PWord(@EndRecord[10])^;
    RequireCpu((Entries > 0) and (Entries <= 4096) and (CentralEnd > CentralStart) and
      (CentralEnd <= FileStream.Size - SizeOf(EndRecord)), 'Invalid PyTorch ZIP directory');
    FileStream.Position := CentralStart;
    TotalBytes := 0;
    for I := 1 to Entries do
    begin
      RequireCpu(FileStream.Position + SizeOf(Central) <= CentralEnd, 'Truncated PyTorch ZIP directory');
      FileStream.ReadBuffer(Central, SizeOf(Central));
      RequireCpu((PCardinal(@Central[0])^ = $02014B50) and (PWord(@Central[10])^ = 0) and
        ((PWord(@Central[8])^ and 1) = 0) and (PWord(@Central[34])^ = 0),
        'Expected an unencrypted, stored PyTorch ZIP member');
      NameSize := PWord(@Central[28])^;
      MemberSize := PCardinal(@Central[24])^;
      MemberOffset := PCardinal(@Central[42])^;
      NextEntry := FileStream.Position + NameSize + PWord(@Central[30])^ + PWord(@Central[32])^;
      RequireCpu((NameSize > 0) and (NameSize <= 1024) and (NextEntry <= CentralEnd) and
        (PCardinal(@Central[20])^ = MemberSize) and (MemberSize <= MaximumArchive - TotalBytes) and
        (MemberOffset + SizeOf(Local) <= CentralStart), 'Invalid PyTorch ZIP member');
      SetLength(Name, NameSize);
      FileStream.ReadBuffer(Name[1], NameSize);
      MemberName := UTF8Decode(Name);
      RequireCpu(not FArchive.ContainsKey(MemberName), 'Duplicate PyTorch ZIP member');
      FileStream.Position := MemberOffset;
      FileStream.ReadBuffer(Local, SizeOf(Local));
      RequireCpu((PCardinal(@Local[0])^ = $04034B50) and (PWord(@Local[8])^ = 0) and
        (PWord(@Local[26])^ = NameSize) and
        (FileStream.Position + NameSize + PWord(@Local[28])^ + MemberSize <= CentralStart),
        'Invalid PyTorch ZIP local header');
      SetLength(LocalName, NameSize);
      FileStream.ReadBuffer(LocalName[1], NameSize);
      RequireCpu(Name = LocalName, 'PyTorch ZIP member name differs from its directory');
      Inc(MemberOffset, SizeOf(Local) + NameSize + PWord(@Local[28])^);
      FileStream.Position := MemberOffset;
      Mem := TMemoryStream.Create;
      try
        if MemberSize > 0 then Mem.CopyFrom(FileStream, MemberSize);
        Mem.Position := 0;
        FArchive.Add(MemberName, Mem);
        Mem := nil;
      finally
        Mem.Free;
      end;
      Inc(TotalBytes, MemberSize);
      FileStream.Position := NextEntry;
    end;
    RequireCpu(FileStream.Position = CentralEnd, 'Unexpected bytes in the PyTorch ZIP directory');
  finally
    FileStream.Free;
  end;
  for MemberName in FArchive.Keys do
    if Copy(MemberName, Length(MemberName) - 8, 9) = '/data.pkl' then
    begin
      RequireCpu(FPickle = nil, 'PyTorch archive has more than one object stream');
      FPrefix := Copy(MemberName, 1, Length(MemberName) - 8);
      FPickle := FArchive[MemberName];
    end;
  RequireCpu((FPickle <> nil) and (FPickle.Size > 0) and (FPickle.Size <= 16 * 1024 * 1024),
    'Missing PyTorch object stream');
end;

constructor TTorchCheckpoint.Create(const FileName: string);
var
  Root, Model: TPickleValue;
begin
  inherited Create;
  FArchive := TObjectDictionary<string, TMemoryStream>.Create([doOwnsValues]);
  FObjects := TObjectList<TPickleValue>.Create(True);
  FStack := TList<TPickleValue>.Create;
  FMemo := TDictionary<Cardinal, TPickleValue>.Create;
  FTensors := TDictionary<string, TTensor>.Create;
  FEpsilon := TDictionary<string, Single>.Create;
  ReadArchive(FileName);
  Root := ReadPickle;
  Model := Field(Root, 'ema');
  if (Model = nil) or (Model.Kind = pvNone) then Model := Field(Root, 'model');
  ReadModule(Model, '', 0);
  RequireCpu(FTensors.Count > 0, 'PyTorch archive has no model parameters');
  FreeAndNil(FMemo);
  FreeAndNil(FStack);
  FreeAndNil(FObjects);
end;

destructor TTorchCheckpoint.Destroy;
begin
  FEpsilon.Free;
  FTensors.Free;
  FMemo.Free;
  FStack.Free;
  FObjects.Free;
  FArchive.Free;
  inherited;
end;

function TTorchCheckpoint.NewValue: TPickleValue;
begin
  RequireCpu(FObjects.Count < 200000, 'PyTorch object stream contains too many values');
  Result := TPickleValue.Create;
  FObjects.Add(Result);
end;

function TTorchCheckpoint.Pop: TPickleValue;
begin
  RequireCpu(FStack.Count > 0, 'Invalid PyTorch object stack');
  Result := FStack.Last;
  FStack.Delete(FStack.Count - 1);
end;

function TTorchCheckpoint.Top: TPickleValue;
begin
  RequireCpu((FStack.Count > 0) and (FStack.Last <> nil), 'Missing PyTorch object');
  Result := FStack.Last;
end;

function TTorchCheckpoint.PopItems(const Count: Integer): TArray<TPickleValue>;
var
  I: Integer;
begin
  RequireCpu((Count >= 0) and (Count <= FStack.Count), 'Invalid PyTorch tuple size');
  SetLength(Result, Count);
  for I := Count - 1 downto 0 do
  begin
    Result[I] := Pop;
    RequireCpu(Result[I] <> nil, 'Unexpected PyTorch object marker');
  end;
end;

function TTorchCheckpoint.PopMarked: TArray<TPickleValue>;
var
  I: Integer;
begin
  I := FStack.Count - 1;
  while (I >= 0) and (FStack[I] <> nil) do Dec(I);
  RequireCpu(I >= 0, 'Missing PyTorch object marker');
  Result := PopItems(FStack.Count - I - 1);
  Pop;
end;

function TTorchCheckpoint.ReadByte: Byte;
begin
  FPickle.ReadBuffer(Result, SizeOf(Result));
end;

function TTorchCheckpoint.ReadCardinal: Cardinal;
begin
  FPickle.ReadBuffer(Result, SizeOf(Result));
end;

function TTorchCheckpoint.ReadLine: string;
var
  Symbol: Byte;
begin
  Result := '';
  repeat
    Symbol := ReadByte;
    if Symbol = 10 then Exit;
    RequireCpu((Length(Result) < 512) and (Symbol >= 32) and (Symbol <= 126), 'Invalid PyTorch symbol');
    Result := Result + Char(Symbol);
  until False;
end;

function TTorchCheckpoint.ReadText(const Count: Integer): string;
var
  Raw: UTF8String;
begin
  RequireCpu((Count >= 0) and (Count <= 1024 * 1024) and (Count <= FPickle.Size - FPickle.Position),
    'Invalid PyTorch string length');
  SetLength(Raw, Count);
  if Count > 0 then FPickle.ReadBuffer(Raw[1], Count);
  Result := UTF8Decode(Raw);
end;

function TTorchCheckpoint.ReadStorage(const Identity: TPickleValue): TPickleValue;
begin
  RequireCpu(
    (Identity <> nil) and (Identity.Kind = pvSequence) and (Length(Identity.Items) >= 5) and (Identity.Items[0].Text = 'storage') and (Identity.Items[1].Kind = pvGlobal) and (Identity.Items[2].Kind = pvText) and (Identity.Items[4].Kind = pvNumber),
    'Unsupported PyTorch persistent object'
  );
  Result := NewValue;
  Result.Kind := pvStorage;
  Result.Storage.DataType := Identity.Items[1].Text;
  Result.Storage.Key := Identity.Items[2].Text;
  Result.Storage.Count := Identity.Items[4].Number;
end;

function TTorchCheckpoint.Rebuild(const Callable, Arguments: TPickleValue): TPickleValue;
var
  I: Integer;
begin
  RequireCpu((Callable <> nil) and (Arguments <> nil) and (Arguments.Kind = pvSequence),
    'Invalid PyTorch constructor record');
  if Callable.Text = 'torch._utils _rebuild_parameter' then
  begin
    RequireCpu((Length(Arguments.Items) >= 1) and (Arguments.Items[0].Kind = pvTensor),
      'Invalid PyTorch parameter record');
    Exit(Arguments.Items[0]);
  end;
  if Callable.Text = 'torch Size' then Exit(Arguments);
  Result := NewValue;
  if Callable.Text = 'collections OrderedDict' then
  begin
    RequireCpu(Length(Arguments.Items) = 0, 'Unsupported PyTorch OrderedDict constructor');
    Result.Kind := pvMap;
    Result.Map := TDictionary<string, TPickleValue>.Create;
  end
  else if Callable.Text = 'torch._utils _rebuild_tensor_v2' then
    begin
      RequireCpu((Length(Arguments.Items) >= 4) and (Arguments.Items[0].Kind = pvStorage) and
        (Arguments.Items[1].Kind = pvNumber) and (Arguments.Items[2].Kind = pvSequence) and
        (Arguments.Items[3].Kind = pvSequence) and
        (Length(Arguments.Items[2].Items) = Length(Arguments.Items[3].Items)),
        'Invalid PyTorch tensor record');
      Result.Kind := pvTensor;
      Result.Tensor.Storage := Arguments.Items[0].Storage;
      Result.Tensor.Offset := Arguments.Items[1].Number;
      SetLength(Result.Tensor.Shape, Length(Arguments.Items[2].Items));
      SetLength(Result.Tensor.Strides, Length(Result.Tensor.Shape));
      for I := 0 to High(Result.Tensor.Shape) do
      begin
        Result.Tensor.Shape[I] := Arguments.Items[2].Items[I].Number;
        Result.Tensor.Strides[I] := Arguments.Items[3].Items[I].Number;
      end;
    end
    else begin
      Result.Kind := pvObject;
      Result.Text := Callable.Text;
    end;
end;

function TTorchCheckpoint.ReadPickle: TPickleValue;
var
  Opcode: Byte;
  Value, Current, Other: TPickleValue;
  Items: TArray<TPickleValue>;
  Key: Cardinal;
  Small: Word;
  Bits: UInt64;
  I: Integer;

  procedure SetPairs(const Target: TPickleValue; const Pairs: TArray<TPickleValue>);
  var
    At: Integer;
  begin
    RequireCpu((Target.Kind = pvMap) and not Odd(Length(Pairs)), 'Invalid PyTorch mapping');
    At := 0;
    while At < Length(Pairs) do
    begin
      if Pairs[At].Kind = pvText then Target.Map.AddOrSetValue(Pairs[At].Text, Pairs[At + 1]);
      Inc(At, 2);
    end;
  end;

begin
  repeat
    Opcode := ReadByte;
    case Opcode of
      $80: RequireCpu(ReadByte = 2, 'Only PyTorch pickle protocol 2 is supported');
      Ord('.'):
        begin
          Result := Pop;
          RequireCpu((Result <> nil) and (FStack.Count = 0), 'Invalid PyTorch root object');
          Exit;
        end;
      Ord('('): FStack.Add(nil);
      Ord('q'), Ord('r'):
        begin
          if Opcode = Ord('q') then
            Key := ReadByte
          else
            Key := ReadCardinal;
          FMemo.AddOrSetValue(Key, Top);
        end;
      Ord('h'), Ord('j'):
        begin
          if Opcode = Ord('h') then
            Key := ReadByte
          else
            Key := ReadCardinal;
          RequireCpu(FMemo.TryGetValue(Key, Value), 'Invalid PyTorch memo reference');
          FStack.Add(Value);
        end;
      Ord('N'), $88, $89, Ord('K'), Ord('M'), Ord('J'), Ord('G'), Ord('X'), Ord('c'):
        begin
          Value := NewValue;
          case Opcode of
            Ord('N'): Value.Kind := pvNone;
            $88, $89, Ord('K'), Ord('M'), Ord('J'):
              begin
                Value.Kind := pvNumber;
                case Opcode of
                  $88: Value.Number := 1;
                  $89: Value.Number := 0;
                  Ord('K'): Value.Number := ReadByte;
                  Ord('M'):
                    begin
                      FPickle.ReadBuffer(Small, SizeOf(Small));
                      Value.Number := Small;
                    end;
                  Ord('J'): Value.Number := Int32(ReadCardinal);
                end;
              end;
            Ord('G'):
              begin
                Value.Kind := pvFloat;
                Bits := 0;
                for I := 1 to 8 do Bits := (Bits shl 8) or ReadByte;
                Move(Bits, Value.Scalar, SizeOf(Bits));
              end;
            Ord('X'):
              begin
                Value.Kind := pvText;
                Value.Text := ReadText(ReadCardinal);
              end;
            Ord('c'):
              begin
                Value.Kind := pvGlobal;
                Value.Text := ReadLine + ' ' + ReadLine;
              end;
          end;
          FStack.Add(Value);
        end;
      Ord(')'), Ord(']'), Ord('t'), $85, $86, $87, Ord('}'):
        begin
          Value := NewValue;
          if Opcode = Ord('}') then
          begin
            Value.Kind := pvMap;
            Value.Map := TDictionary<string, TPickleValue>.Create;
          end
          else begin
            Value.Kind := pvSequence;
            if Opcode = Ord('t') then
              Value.Items := PopMarked
            else if Opcode in [$85, $86, $87] then
                Value.Items := PopItems(Opcode - $84);
          end;
          FStack.Add(Value);
        end;
      Ord('s'):
        begin
          Items := PopItems(2); SetPairs(Top, Items);
        end;
      Ord('u'):
        begin
          Items := PopMarked; SetPairs(Top, Items);
        end;
      Ord('a'), Ord('e'):
        begin
          if Opcode = Ord('a') then
            Items := PopItems(1)
          else
            Items := PopMarked;
          Current := Top;
          RequireCpu(Current.Kind = pvSequence, 'Invalid PyTorch list');
          Current.Items := Current.Items + Items;
        end;
      Ord('Q'): FStack.Add(ReadStorage(Pop));
      Ord('R'):
        begin
          Other := Pop;
          Current := Pop;
          FStack.Add(Rebuild(Current, Other));
        end;
      $81:
        begin
          Pop;
          Current := Pop;
          Value := NewValue;
          Value.Kind := pvObject;
          Value.Text := Current.Text;
          FStack.Add(Value);
        end;
      Ord('b'):
        begin
          Other := Pop;
          Current := Top;
          Current.State := Other;
        end;
    else
      raise ECpuGraph.CreateFmt('Unsupported PyTorch object opcode 0x%.2x', [Opcode]);
    end;
  until False;
end;

function TTorchCheckpoint.Field(const Map: TPickleValue; const Name: string): TPickleValue;
begin
  Result := nil;
  if (Map <> nil) and (Map.Map <> nil) then Map.Map.TryGetValue(Name, Result);
end;

procedure TTorchCheckpoint.ReadModule(const Model: TPickleValue; const Name: string; const Depth: Integer);
const
  TensorFields: array[0..1] of string = ('_parameters', '_buffers');
var
  State, Values, Child: TPickleValue;
  Part, Prefix, Key: string;
begin
  RequireCpu((Depth <= 64) and (Model <> nil) and (Model.Kind = pvObject), 'Invalid PyTorch model module');
  State := Model.State;
  RequireCpu((State <> nil) and (State.Kind = pvMap), 'Missing PyTorch model module state');
  Prefix := Name;
  if Prefix <> '' then Prefix := Prefix + '.';
  Child := Field(State, 'eps');
  if (Child <> nil) and (Child.Kind = pvFloat) then FEpsilon.Add(Name, Child.Scalar);
  for Part in TensorFields do
  begin
    Values := Field(State, Part);
    if (Values = nil) or (Values.Map = nil) then Continue;
    for Key in Values.Map.Keys do
    begin
      Child := Values.Map[Key];
      if Child.Kind = pvTensor then FTensors.Add(Prefix + Key, Child.Tensor);
    end;
  end;
  Values := Field(State, '_modules');
  if (Values <> nil) and (Values.Map <> nil) then
    for Key in Values.Map.Keys do
    begin
      Child := Values.Map[Key];
      if Child.Kind <> pvNone then ReadModule(Child, Prefix + Key, Depth + 1);
    end;
end;

function TTorchCheckpoint.Contains(const Name: string): Boolean;
begin
  Result := FTensors.ContainsKey(Name);
end;

function TTorchCheckpoint.ReadFloats(const Name: string; out Shape: TArray<Int64>): TArray<Single>;
var
  Tensor: TTensor;
  Storage: TMemoryStream;
  Count, Width, Step: Int64;
  Half: TFloat16;
  I: Integer;
  Data: Pointer;
begin
  RequireCpu(FTensors.TryGetValue(Name, Tensor), 'PyTorch tensor is missing: ' + Name);
  Shape := Tensor.Shape;
  Count := TensorElements(Shape);
  Step := 1;
  for I := High(Shape) downto 0 do
  begin
    RequireCpu((Shape[I] = 1) or (Tensor.Strides[I] = Step),
      'Non-contiguous PyTorch weight: ' + Name);
    Step := Step * Shape[I];
  end;
  Width := 0;
  if Tensor.Storage.DataType = 'torch HalfStorage' then Width := 2;
  if Tensor.Storage.DataType = 'torch FloatStorage' then Width := 4;
  RequireCpu((Width > 0) and (Count > 0) and (Count <= MaximumArchive div 4) and
    (Tensor.Offset >= 0) and (Tensor.Offset <= Tensor.Storage.Count - Count) and
    FArchive.TryGetValue(FPrefix + 'data/' + Tensor.Storage.Key, Storage) and
    (Storage.Size div Width = Tensor.Storage.Count) and (Storage.Size mod Width = 0),
    'Invalid PyTorch floating-point storage: ' + Name);
  Data := PByte(Storage.Memory) + Tensor.Offset * Width;
  SetLength(Result, Count);
  if Width = 4 then
    Move(Data^, Result[0], Count * Width)
  else
    for I := 0 to High(Result) do
    begin
      Half.Value := PWord(Data)[I];
      Result[I] := Half.ToSingle;
    end;
end;

function TTorchCheckpoint.BatchNormEpsilon(const Name: string): Single;
begin
  RequireCpu(FEpsilon.TryGetValue(Name, Result) and (Result > 0) and (Result < 1),
    'Missing batch-norm epsilon: ' + Name);
end;

function TTorchCheckpoint.TensorNames: TArray<string>;
begin
  Result := FTensors.Keys.ToArray;
end;

end.
