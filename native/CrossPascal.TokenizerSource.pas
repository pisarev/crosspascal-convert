{ ************************************************************************** }
{                                                                            }
{ CrossPascal.TokenizerSource                                                }
{                                                                            }
{ Copyright © 2026 Yuriy Pisarev                                             }
{                                                                            }
{ ************************************************************************** }

unit CrossPascal.TokenizerSource;

{$IFDEF FPC}{$MODE DELPHIUNICODE}{$ENDIF}

interface

procedure ExportGemmaTokenizer(const SourceFile, Destination: string; const VocabularySize: Integer);

implementation

uses
  {$IFDEF FPC}SysUtils, Classes, Generics.Collections{$ELSE}System.SysUtils, System.Classes, System.Generics.Collections{$ENDIF},
  CrossPascal.UnicodeJson, TVM.CpuGraph;

function ByteToken(const Text: string; out Value: Integer): Boolean;
begin
  Result := (Length(Text) = 6) and (Copy(Text, 1, 3) = '<0x') and (Text[6] = '>');
  if Result then
    RequireCpu(TryStrToInt('$' + Copy(Text, 4, 2), Value) and (Value in [0..255]), 'Invalid byte token');
end;

function CodePoint(const Text: string): Integer;
begin
  Result := -1;
  if Length(Text) = 1 then Exit(Ord(Text[1]));
  if (Length(Text) = 2) and (Ord(Text[1]) >= $D800) and (Ord(Text[1]) <= $DBFF) and
    (Ord(Text[2]) >= $DC00) and (Ord(Text[2]) <= $DFFF) then
      Result := $10000 + (Ord(Text[1]) - $D800) * 1024 + Ord(Text[2]) - $DC00;
end;

procedure WriteText(const Stream: TStream; const Value: string);
var
  UTF8: UTF8String;
begin
  UTF8 := TokenizerUTF8(Value);
  if Length(UTF8) > 0 then Stream.WriteBuffer(Pointer(UTF8)^, Length(UTF8));
end;

procedure ExportGemmaTokenizer(const SourceFile, Destination: string; const VocabularySize: Integer);
var
  Reader: TUnicodeJsonReader;
  Words: TArray<string>;
  Lookup: TDictionary<string, Integer>;
  CharacterIDs: TArray<Integer>;
  ByteIDs: array[0..255] of Int32;
  Pieces, Index, Rules, Characters, Added, Metadata: TMemoryStream;
  IndexEntry: array[0..1] of UInt32;
  Rule: array[0..2] of Int32;
  CharEntry: array[0..1] of Int32;
  Name, Left, Right, Text, SpecialLines: string;
  Encoded: UTF8String;
  I, Delimiter, Token, Raw, Point, RuleCount, CharacterCount, BOS, Root, Model,
    Normalizer, Special, SpecialCount, WordCount, Offset: Integer;
  IsSpecial: Boolean;
  ByteValue: Byte;
begin
  RequireCpu((VocabularySize > 0) and DirectoryExists(Destination), 'Invalid Gemma tokenizer destination');
  Reader := TUnicodeJsonReader.Create(SourceFile);
  Lookup := nil;
  Pieces := nil;
  Index := nil;
  Rules := nil;
  Characters := nil;
  Added := nil;
  Metadata := nil;
  try
    Root := Reader.Position;
    Reader.SeekMember('normalizer');
    Normalizer := Reader.Position;
    Reader.SeekMember('type');
    RequireCpu(Reader.ReadString = 'Replace', 'Unsupported Gemma whitespace normalization');
    Reader.Position := Normalizer;
    Reader.SeekMember('pattern');
    Reader.SeekMember('String');
    RequireCpu(Reader.ReadString = ' ', 'Unsupported Gemma whitespace pattern');
    Reader.Position := Normalizer;
    Reader.SeekMember('content');
    RequireCpu(Reader.ReadString = #$2581, 'Unsupported Gemma whitespace replacement');
    Reader.Position := Root;
    Reader.SeekMember('model');
    Model := Reader.Position;
    Reader.SeekMember('type');
    RequireCpu(Reader.ReadString = 'BPE', 'Expected a Gemma BPE tokenizer');
    Reader.Position := Model;
    Reader.SeekMember('byte_fallback');
    RequireCpu(Reader.ReadBoolean, 'Expected Gemma byte fallback');
    SetLength(Words, VocabularySize);
    SetLength(CharacterIDs, $110000);
    for I := 0 to High(CharacterIDs) do CharacterIDs[I] := -1;
    for I := Low(ByteIDs) to High(ByteIDs) do ByteIDs[I] := -1;
    Lookup := TDictionary<string, Integer>.Create(VocabularySize);
    Reader.Position := Model;
    Reader.SeekMember('vocab');
    Reader.Expect('{');
    WordCount := 0;
    if not Reader.Take('}') then
      repeat
        Name := Reader.ReadMember;
        Token := Reader.ReadInteger;
        RequireCpu((Name <> '') and (Token >= 0) and (Token < VocabularySize) and (Words[Token] = ''),
          'Duplicate or invalid Gemma vocabulary ID');
        Words[Token] := Name;
        if Lookup.TryGetValue(Name, Raw) then
          raise EConvertError.CreateFmt('Gemma vocabulary IDs %d (%x) and %d (%x) have the same token',
            [Raw, CodePoint(Words[Raw]), Token, CodePoint(Name)]);
        Lookup.Add(Name, Token);
        if ByteToken(Name, Raw) then ByteIDs[Raw] := Token;
        Point := CodePoint(Name);
        if Point >= 0 then CharacterIDs[Point] := Token;
        Inc(WordCount);
      until not Reader.More('}');
    RequireCpu(WordCount = VocabularySize, 'Gemma tokenizer vocabulary size differs from the checkpoint');
    Pieces := TMemoryStream.Create;
    Index := TMemoryStream.Create;
    Rules := TMemoryStream.Create;
    Characters := TMemoryStream.Create;
    Added := TMemoryStream.Create;
    Metadata := TMemoryStream.Create;
    for I := 0 to High(Words) do
    begin
      Name := Words[I];
      RequireCpu((Name <> '') and (Pieces.Position <= High(UInt32)),
        'Gemma token bytes are missing or too large');
      IndexEntry[0] := Pieces.Position;
      if ByteToken(Name, Raw) then
      begin
        ByteValue := Raw;
        Pieces.WriteBuffer(ByteValue, SizeOf(ByteValue));
        IndexEntry[1] := 1;
      end
      else begin
        Encoded := TokenizerUTF8(Name);
        Pieces.WriteBuffer(Pointer(Encoded)^, Length(Encoded));
        IndexEntry[1] := Length(Encoded);
      end;
      Index.WriteBuffer(IndexEntry, SizeOf(IndexEntry));
    end;
    RuleCount := 0;
    Reader.Position := Model;
    Reader.SeekMember('merges');
    Reader.Expect('[');
    if not Reader.Take(']') then
      repeat
        if Reader.Take('[') then
        begin
          Left := Reader.ReadString;
          Reader.Expect(',');
          Right := Reader.ReadString;
          Reader.Expect(']');
        end
        else begin
          Text := Reader.ReadString;
          Delimiter := Pos(' ', Text);
          RequireCpu(Delimiter > 0, 'Invalid Gemma BPE merge line');
          Left := Copy(Text, 1, Delimiter - 1);
          Right := Copy(Text, Delimiter + 1, MaxInt);
        end;
        if Lookup.TryGetValue(Left, Rule[0]) and Lookup.TryGetValue(Right, Rule[1]) and
          Lookup.TryGetValue(Left + Right, Rule[2]) then
          begin
            Rules.WriteBuffer(Rule, SizeOf(Rule));
            Inc(RuleCount);
          end;
      until not Reader.More(']');
    CharacterCount := 0;
    for Point := 0 to High(CharacterIDs) do
      if CharacterIDs[Point] >= 0 then
      begin
        CharEntry[0] := Point;
        CharEntry[1] := CharacterIDs[Point];
        Characters.WriteBuffer(CharEntry, SizeOf(CharEntry));
        Inc(CharacterCount);
      end;
    SpecialCount := 0;
    SpecialLines := '';
    Reader.Position := Root;
    Reader.SeekMember('added_tokens');
    Reader.Expect('[');
    if not Reader.Take(']') then
      repeat
        Special := Reader.Position;
        Reader.SeekMember('id');
        Token := Reader.ReadInteger;
        Reader.Position := Special;
        Reader.SeekMember('content');
        Name := Reader.ReadString;
        for I := Length(Name) downto 1 do
          if Name[I] = #10 then Name := Copy(Name, 1, I - 1) + '\n' + Copy(Name, I + 1, MaxInt);
        Reader.Position := Special;
        Reader.SeekMember('special');
        IsSpecial := Reader.ReadBoolean;
        RequireCpu((Token >= 0) and (Token < VocabularySize), 'Invalid Gemma added token ID');
        SpecialLines := SpecialLines + IntToStr(Token) + ' ' + IntToStr(Ord(IsSpecial)) + ' ' + Name + #10;
        Inc(SpecialCount);
        Reader.Position := Special;
        Reader.Skip;
      until not Reader.More(']');
    WriteText(Added, IntToStr(SpecialCount) + #10 + SpecialLines);
    BOS := -1;
    Reader.Position := Root;
    Offset := Reader.FindMember('post_processor');
    if Offset <> 0 then
    begin
      Reader.Position := Offset;
      if Reader.Peek = '{' then
      begin
        Reader.SeekMember('special_tokens');
        Offset := Reader.FindMember('<bos>');
        if Offset <> 0 then
        begin
          Reader.Position := Offset;
          Reader.SeekMember('ids');
          Reader.Expect('[');
          if not Reader.Take(']') then BOS := Reader.ReadInteger;
        end;
      end;
    end;
    WriteText(Metadata, 'style sentencepiece'#10'tokens ' + IntToStr(VocabularySize) + #10 +
      'merges ' + IntToStr(RuleCount) + #10 + 'chars ' + IntToStr(CharacterCount) + #10 +
      'specials ' + IntToStr(SpecialCount) + #10 + 'bytes ' + IntToStr(Pieces.Size) + #10 +
      'bos ' + IntToStr(BOS) + #10 + 'space_meta 2581'#10'normalizer Replace'#10);
    Pieces.SaveToFile(BundleMember(Destination, 'tokenizer-tokens.bin'));
    Index.SaveToFile(BundleMember(Destination, 'tokenizer-tokens.index'));
    Rules.SaveToFile(BundleMember(Destination, 'tokenizer-merges.bin'));
    Characters.SaveToFile(BundleMember(Destination, 'tokenizer-chars.bin'));
    Added.SaveToFile(BundleMember(Destination, 'tokenizer-specials.txt'));
    Metadata.SaveToFile(BundleMember(Destination, 'tokenizer.index'));
    Pieces.Clear;
    Pieces.WriteBuffer(ByteIDs, SizeOf(ByteIDs));
    Pieces.SaveToFile(BundleMember(Destination, 'tokenizer-bytes.bin'));
  finally
    Metadata.Free;
    Added.Free;
    Characters.Free;
    Rules.Free;
    Index.Free;
    Pieces.Free;
    Lookup.Free;
    Reader.Free;
  end;
end;

end.
