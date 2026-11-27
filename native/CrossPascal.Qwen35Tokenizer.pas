{ ************************************************************************** }
{                                                                            }
{ CrossPascal.Qwen35Tokenizer                                                 }
{                                                                            }
{ Copyright © 2026 Yuriy Pisarev                                              }
{                                                                            }
{ ************************************************************************** }

unit CrossPascal.Qwen35Tokenizer;

{$IFDEF FPC}{$MODE DELPHIUNICODE}{$ENDIF}

interface

uses
  CrossPascal.GGUF;

{ Reconstructs the dense Qwen 3.5/3.8 byte-level tokenizer from its own GGUF
  vocabulary/types/merges. No second checkpoint or Internet access is needed.
  Returns the effective vocabulary, excluding only a trailing UNUSED region. }
function ExportQwen35GGUFTokenizer(const Source: TGGUFCheckpoint;
  const DestinationFile: string): Integer;

implementation

uses
  {$IFDEF FPC}SysUtils, Classes{$ELSE}System.SysUtils, System.Classes{$ENDIF},
  CrossPascal.Json, TVM.CpuGraph, CrossPascal.NativeIO;

function ExportQwen35GGUFTokenizer(const Source: TGGUFCheckpoint;
  const DestinationFile: string): Integer;
const
  Pattern: string = '(?i:''s|''t|''re|''ve|''m|''ll|''d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+';
var
  Pieces, Merges: TArray<UTF8String>;
  Types: TArray<Int32>;
  Root, Model, Row, Pre, Split, Match, ByteLevel, Normalizer, Decoder: TJSONObject;
  Added, Rules, Sequence: TJSONArray;
  I, BaseCount, Tail: Integer;
  Stream: TStream;
  Bytes: UTF8String;
  Serialized: string;

  function IsControl(const Index: Integer): Boolean;
  var
    Piece: UTF8String;
  begin
    Piece := Pieces[Index];
    { GGUF marks the Qwen code separators CONTROL. The dense Qwen3.5/3.8
      tokenizer preserves these six literal code markers when decoding. }
    Result := (Types[Index] = 3) and (Piece <> '<|fim_prefix|>') and
      (Piece <> '<|fim_middle|>') and (Piece <> '<|fim_suffix|>') and
      (Piece <> '<|fim_pad|>') and (Piece <> '<|repo_name|>') and (Piece <> '<|file_sep|>');
  end;

  procedure WriteUTF8(const Value: UTF8String);
  begin
    if Value <> '' then Stream.WriteBuffer(Value[1], Length(Value));
  end;

  function Quoted(const Value: string): UTF8String;
  var
    Scalar: TJSONObject;
    Item: TJSONValue;
  begin
    Scalar := TJSONObject.Create;
    try
      Scalar.AddPair('v', Value);
      Item := Scalar.GetValue('v');
      {$IFDEF FPC}Result := UTF8String(Item.AsJSON);
      {$ELSE}Result := UTF8Encode(Item.ToJSON);{$ENDIF}
    finally
      Scalar.Free;
    end;
  end;
begin
  RequireCpu((Source <> nil) and not FileExists(DestinationFile), 'Tokenizer output already exists');
  RequireCpu((Source.Text('general.architecture') = 'qwen35') and
    (Source.Text('tokenizer.ggml.model') = 'gpt2') and
    (Source.Text('tokenizer.ggml.pre') = 'qwen35'), 'Unsupported Qwen GGUF tokenizer metadata');
  Pieces := Source.TokenPieces('tokenizer.ggml.tokens');
  Types := Source.IntArray('tokenizer.ggml.token_type');
  Merges := Source.TokenPieces('tokenizer.ggml.merges');
  RequireCpu((Length(Pieces) = Length(Types)) and (Length(Pieces) >= 256), 'Invalid Qwen vocabulary');
  Result := Length(Pieces);
  while (Result > 0) and (Types[Result - 1] = 5) do Dec(Result);
  BaseCount := 0;
  while (BaseCount < Result) and (Types[BaseCount] = 1) do Inc(BaseCount);
  RequireCpu((BaseCount >= 256) and (BaseCount < Result), 'Qwen base vocabulary or added tokens are absent');
  Root := TJSONObject.Create;
  try
    Root.AddPair('version', '1.0');
    Root.AddPair('truncation', TJSONNull.Create);
    Root.AddPair('padding', TJSONNull.Create);
    Added := TJSONArray.Create;
    Root.AddPair('added_tokens', Added);
    for I := BaseCount to Result - 1 do
    begin
      RequireCpu(Types[I] in [3, 4], 'Noncontiguous Qwen added-token IDs or UNUSED vocabulary holes');
      Row := TJSONObject.Create;
      Added.AddElement(Row);
      Row.AddPair('id', JSONInteger(I));
      Row.AddPair('content', UTF8Decode(Pieces[I]));
      Row.AddPair('single_word', JSONBoolean(False));
      Row.AddPair('lstrip', JSONBoolean(False));
      Row.AddPair('rstrip', JSONBoolean(False));
      Row.AddPair('normalized', JSONBoolean(False));
      Row.AddPair('special', JSONBoolean(IsControl(I)));
    end;
    Normalizer := TJSONObject.Create;
    Normalizer.AddPair('type', 'NFC');
    Root.AddPair('normalizer', Normalizer);
    Pre := TJSONObject.Create;
    Root.AddPair('pre_tokenizer', Pre);
    Pre.AddPair('type', 'Sequence');
    Sequence := TJSONArray.Create;
    Pre.AddPair('pretokenizers', Sequence);
    Split := TJSONObject.Create;
    Sequence.AddElement(Split);
    Split.AddPair('type', 'Split');
    Match := TJSONObject.Create;
    Split.AddPair('pattern', Match);
    Match.AddPair('Regex', Pattern);
    Split.AddPair('behavior', 'Isolated');
    Split.AddPair('invert', JSONBoolean(False));
    ByteLevel := TJSONObject.Create;
    Sequence.AddElement(ByteLevel);
    ByteLevel.AddPair('type', 'ByteLevel');
    ByteLevel.AddPair('add_prefix_space', JSONBoolean(False));
    ByteLevel.AddPair('trim_offsets', JSONBoolean(True));
    ByteLevel.AddPair('use_regex', JSONBoolean(False));
    Root.AddPair('post_processor', TJSONNull.Create);
    Decoder := TJSONObject.Create;
    Decoder.AddPair('type', 'ByteLevel');
    Decoder.AddPair('add_prefix_space', JSONBoolean(False));
    Decoder.AddPair('trim_offsets', JSONBoolean(True));
    Decoder.AddPair('use_regex', JSONBoolean(False));
    Root.AddPair('decoder', Decoder);
    Model := TJSONObject.Create;
    Root.AddPair('model', Model);
    Model.AddPair('type', 'BPE');
    Model.AddPair('dropout', TJSONNull.Create);
    Model.AddPair('unk_token', TJSONNull.Create);
    Model.AddPair('continuing_subword_prefix', TJSONNull.Create);
    Model.AddPair('end_of_word_suffix', TJSONNull.Create);
    Model.AddPair('fuse_unk', JSONBoolean(False));
    Model.AddPair('byte_fallback', JSONBoolean(False));
    Model.AddPair('ignore_merges', JSONBoolean(False));
    Rules := TJSONArray.Create;
    Model.AddPair('merges', Rules);
    for I := 0 to High(Merges) do Rules.AddElement(TJSONString.Create(UTF8Decode(Merges[I])));
    { FPC's JSON object-name hash truncates keys at255 UTF8 bytes. Stream this
      dictionary instead; token strings are data and may exceed that limit. }
    Serialized := Root.ToJSON;
    Tail := Length(Serialized);
    while (Tail > 0) and CharInSet(Serialized[Tail], [#9, #10, #13, ' ']) do Dec(Tail);
    RequireCpu((Tail > 0) and (Serialized[Tail] = '}'), 'Invalid tokenizer root serialization');
    Dec(Tail);
    while (Tail > 0) and CharInSet(Serialized[Tail], [#9, #10, #13, ' ']) do Dec(Tail);
    RequireCpu((Tail > 0) and (Serialized[Tail] = '}'), 'Invalid tokenizer model serialization');
    SetLength(Serialized, Tail - 1);
    Bytes := UTF8Encode(Serialized);
    Stream := TNativeWriteStream.Create(DestinationFile, False);
    try
      WriteUTF8(Bytes);
      WriteUTF8(',"vocab":{');
      for I := 0 to BaseCount - 1 do
      begin
        if I > 0 then WriteUTF8(',');
        WriteUTF8(Quoted(UTF8Decode(Pieces[I])));
        WriteUTF8(':' + UTF8String(IntToStr(I)));
      end;
      WriteUTF8('}}}');
    finally
      Stream.Free;
    end;
  finally
    Root.Free;
  end;
end;

end.
