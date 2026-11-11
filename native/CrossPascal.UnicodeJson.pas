{ ************************************************************************** }
{                                                                            }
{ CrossPascal.UnicodeJson                                                    }
{                                                                            }
{ Copyright © 2026 Yuriy Pisarev (ypisareff@outlook.com)                     }
{                                                                            }
{ ************************************************************************** }

unit CrossPascal.UnicodeJson;

{$IFDEF FPC}{$MODE DELPHIUNICODE}{$ENDIF}

interface

type
  TUnicodeJsonReader = class
  private
    FText: string;
    FPosition: Integer;
    procedure SkipValue(const Depth: Integer);
    function HexDigit: Integer;
  public
    constructor Create(const FileName: string);
    function Peek: Char;
    function Take(const Token: Char): Boolean;
    procedure Expect(const Token: Char);
    function ReadString: string;
    function ReadInteger: Integer;
    function ReadBoolean: Boolean;
    function ReadMember: string;
    procedure Skip;
    function More(const Closing: Char): Boolean;
    function FindMember(const Name: string): Integer;
    procedure SeekMember(const Name: string);
    property Position: Integer read FPosition write FPosition;
  end;

function TokenizerUTF8(const Text: string): UTF8String;

implementation

uses
  {$IFDEF FPC}Classes, SysUtils{$ELSE}System.Classes, System.SysUtils{$ENDIF},
  TVM.CpuGraph;

function DecodeTokenizerUTF8(const Bytes: UTF8String): string;
var
  I, Count, Code, Remaining, Minimum, B: Integer;
begin
  SetLength(Result, Length(Bytes));
  I := 1;
  Count := 0;
  while I <= Length(Bytes) do
  begin
    B := Ord(Bytes[I]);
    Inc(I);
    case B of
      0..$7F:
        begin
          Code := B;
          Remaining := 0;
          Minimum := 0;
        end;
      $C2..$DF:
        begin
          Code := B and $1F;
          Remaining := 1;
          Minimum := $80;
        end;
      $E0..$EF:
        begin
          Code := B and $0F;
          Remaining := 2;
          Minimum := $800;
        end;
      $F0..$F4:
        begin
          Code := B and $07;
          Remaining := 3;
          Minimum := $10000;
        end;
    else
      raise EConvertError.Create('Invalid UTF-8 in tokenizer JSON');
    end;
    RequireCpu(I + Remaining - 1 <= Length(Bytes), 'Truncated UTF-8 in tokenizer JSON');
    while Remaining > 0 do
    begin
      B := Ord(Bytes[I]);
      RequireCpu((B and $C0) = $80, 'Invalid UTF-8 continuation in tokenizer JSON');
      Code := (Code shl 6) or (B and $3F);
      Inc(I);
      Dec(Remaining);
    end;
    RequireCpu((Code >= Minimum) and (Code <= $10FFFF) and
      ((Code < $D800) or (Code > $DFFF)), 'Invalid UTF-8 code point in tokenizer JSON');
    if Code >= $10000 then
    begin
      Dec(Code, $10000);
      Inc(Count);
      Result[Count] := Char($D800 or (Code shr 10));
      Code := $DC00 or (Code and $3FF);
    end;
    Inc(Count);
    Result[Count] := Char(Code);
  end;
  SetLength(Result, Count);
end;

function TokenizerUTF8(const Text: string): UTF8String;
var
  I, Count, Code, LowSurrogate: Integer;

  procedure Append(const Value: Integer);
  begin
    Inc(Count);
    Result[Count] := AnsiChar(Value);
  end;

begin
  SetLength(Result, Length(Text) * 3);
  I := 1;
  Count := 0;
  while I <= Length(Text) do
  begin
    Code := Ord(Text[I]);
    Inc(I);
    if (Code >= $D800) and (Code <= $DFFF) then
    begin
      RequireCpu((Code <= $DBFF) and (I <= Length(Text)), 'Unpaired tokenizer surrogate');
      LowSurrogate := Ord(Text[I]);
      RequireCpu((LowSurrogate >= $DC00) and (LowSurrogate <= $DFFF), 'Unpaired tokenizer surrogate');
      Code := $10000 + (Code - $D800) * 1024 + LowSurrogate - $DC00;
      Inc(I);
    end;
    if Code < $80 then
      Append(Code)
    else begin
      if Code < $800 then
        Append($C0 or (Code shr 6))
      else begin
        if Code < $10000 then
          Append($E0 or (Code shr 12))
        else begin
          Append($F0 or (Code shr 18));
          Append($80 or ((Code shr 12) and $3F));
        end;
        Append($80 or ((Code shr 6) and $3F));
      end;
      Append($80 or (Code and $3F));
    end;
  end;
  SetLength(Result, Count);
end;

constructor TUnicodeJsonReader.Create(const FileName: string);
var
  Stream: TFileStream;
  Bytes: UTF8String;
begin
  inherited Create;
  Stream := TFileStream.Create(FileName, fmOpenRead or fmShareDenyWrite);
  try
    RequireCpu((Stream.Size > 0) and (Stream.Size <= 64 * 1024 * 1024),
      'Invalid tokenizer JSON size');
    SetLength(Bytes, Stream.Size);
    Stream.ReadBuffer(Bytes[1], Length(Bytes));
    FText := DecodeTokenizerUTF8(Bytes);
  finally
    Stream.Free;
  end;
  FPosition := 1;
  if FText[FPosition] = #$FEFF then Inc(FPosition);
end;

function TUnicodeJsonReader.Peek: Char;
begin
  while (FPosition <= Length(FText)) and CharInSet(FText[FPosition], [#9, #10, #13, ' ']) do
    Inc(FPosition);
  RequireCpu(FPosition <= Length(FText), 'Unexpected end of tokenizer JSON');
  Result := FText[FPosition];
end;

function TUnicodeJsonReader.Take(const Token: Char): Boolean;
begin
  Result := Peek = Token;
  if Result then Inc(FPosition);
end;

procedure TUnicodeJsonReader.Expect(const Token: Char);
begin
  RequireCpu(Take(Token), 'Expected "' + Token + '" in tokenizer JSON at ' + IntToStr(FPosition));
end;

function TUnicodeJsonReader.HexDigit: Integer;
var
  Symbol: Char;
begin
  RequireCpu(FPosition <= Length(FText), 'Truncated Unicode escape');
  Symbol := FText[FPosition];
  Inc(FPosition);
  case Symbol of
    '0'..'9': Result := Ord(Symbol) - Ord('0');
    'a'..'f': Result := Ord(Symbol) - Ord('a') + 10;
    'A'..'F': Result := Ord(Symbol) - Ord('A') + 10;
  else
    raise EConvertError.Create('Invalid Unicode escape in tokenizer JSON');
  end;
end;

function TUnicodeJsonReader.ReadString: string;
var
  Start, Code, I: Integer;
  Symbol: Char;
begin
  Expect('"');
  Result := '';
  Start := FPosition;
  while FPosition <= Length(FText) do
  begin
    Symbol := FText[FPosition];
    if Symbol = '"' then
    begin
      Result := Result + Copy(FText, Start, FPosition - Start);
      Inc(FPosition);
      Exit;
    end;
    RequireCpu(Symbol >= #32, 'Control character in tokenizer JSON string');
    if Symbol = '\' then
    begin
      Result := Result + Copy(FText, Start, FPosition - Start);
      Inc(FPosition);
      RequireCpu(FPosition <= Length(FText), 'Truncated tokenizer JSON escape');
      Symbol := FText[FPosition];
      Inc(FPosition);
      case Symbol of
        '"', '\', '/':;
        'b': Symbol := #8;
        'f': Symbol := #12;
        'n': Symbol := #10;
        'r': Symbol := #13;
        't': Symbol := #9;
        'u':
        begin
          Code := 0;
          for I := 1 to 4 do Code := Code * 16 + HexDigit;
          Symbol := Char(Code);
        end;
        else
          raise EConvertError.Create('Invalid tokenizer JSON escape');
      end;
      Result := Result + Symbol;
      Start := FPosition;
    end
    else
      Inc(FPosition);
  end;
  raise EConvertError.Create('Unterminated tokenizer JSON string');
end;

function TUnicodeJsonReader.ReadInteger: Integer;
var
  Start: Integer;
begin
  Peek;
  Start := FPosition;
  if FText[FPosition] = '-' then Inc(FPosition);
  while (FPosition <= Length(FText)) and CharInSet(FText[FPosition], ['0'..'9']) do
    Inc(FPosition);
  RequireCpu(TryStrToInt(Copy(FText, Start, FPosition - Start), Result), 'Invalid tokenizer JSON integer');
end;

function TUnicodeJsonReader.ReadBoolean: Boolean;
begin
  Peek;
  Result := Copy(FText, FPosition, 4) = 'true';
  if Result then
    Inc(FPosition, 4)
  else begin
    RequireCpu(Copy(FText, FPosition, 5) = 'false', 'Invalid tokenizer JSON boolean');
    Inc(FPosition, 5);
  end;
end;

function TUnicodeJsonReader.ReadMember: string;
begin
  Result := ReadString;
  Expect(':');
end;

function TUnicodeJsonReader.More(const Closing: Char): Boolean;
begin
  Result := not Take(Closing);
  if Result then Expect(',');
end;

procedure TUnicodeJsonReader.SkipValue(const Depth: Integer);
var
  Start: Integer;
begin
  RequireCpu(Depth <= 64, 'Tokenizer JSON is nested too deeply');
  case Peek of
    '{':
      begin
        Expect('{');
        if not Take('}') then
          repeat
            ReadMember;
            SkipValue(Depth + 1);
          until not More('}');
      end;
    '[':
      begin
        Expect('[');
        if not Take(']') then
          repeat
            SkipValue(Depth + 1);
          until not More(']');
      end;
    '"': ReadString;
  else begin
      Start := FPosition;
      while (FPosition <= Length(FText)) and
        not CharInSet(FText[FPosition], [#9, #10, #13, ' ', ',', '}', ']']) do Inc(FPosition);
      RequireCpu(FPosition > Start, 'Missing tokenizer JSON value');
    end;
  end;
end;

procedure TUnicodeJsonReader.Skip;
begin
  SkipValue(0);
end;

function TUnicodeJsonReader.FindMember(const Name: string): Integer;
var
  Saved: Integer;
begin
  Saved := FPosition;
  try
    Expect('{');
    if not Take('}') then
      repeat
        if ReadMember = Name then Exit(FPosition);
        SkipValue(0);
      until not More('}');
    Result := 0;
  finally
    FPosition := Saved;
  end;
end;

procedure TUnicodeJsonReader.SeekMember(const Name: string);
var
  Offset: Integer;
begin
  Offset := FindMember(Name);
  RequireCpu(Offset <> 0, 'Missing tokenizer JSON member: ' + Name);
  FPosition := Offset;
end;

end.
