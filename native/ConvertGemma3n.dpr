{ ************************************************************************** }
{                                                                            }
{ ConvertGemma3n                                                             }
{                                                                            }
{ Copyright © 2026 Yuriy Pisarev (ypisareff@outlook.com)                     }
{                                                                            }
{ ************************************************************************** }

program ConvertGemma3n;

{$APPTYPE CONSOLE}
{$IFDEF FPC}{$MODE DELPHIUNICODE}{$ENDIF}

uses
  {$IFDEF FPC}SysUtils{$ELSE}System.SysUtils{$ENDIF}, CrossPascal.TVMCompiler, CrossPascal.Gemma3nBundle;

var
  Kit: TWindowsTvmKit;
  Projector, Nvrtc: string;
  EffectiveCount, Context, Chunk: Integer;

procedure ReadKit(const First: Integer);
begin
  Kit.CompilerDirectory := ParamStr(First);
  Kit.Linker := ParamStr(First + 1);
  Kit.FfiLibrary := ExpandFileName(ParamStr(First + 2));
  Kit.MsvcLibraries := ParamStr(First + 3);
  Kit.UcrtLibraries := ParamStr(First + 4);
  Kit.WindowsLibraries := ParamStr(First + 5);
end;

begin
  if ParamStr(1) = '--attach-vision' then
  begin
    if not (ParamCount in [13, 14]) then
    begin
      WriteLn(ErrOutput,
        'Usage: ConvertGemma3n --attach-vision text-bundle projector.gguf recipe-dir destination');
      WriteLn(ErrOutput,
        '       ggml-base.dll sentencepiece.dll TVM-compiler-directory link.exe tvm_ffi.lib');
      WriteLn(ErrOutput, '       msvc-lib ucrt-lib windows-lib [nvrtc.dll]');
      Halt(2);
    end;
    try
      ReadKit(8);
      Nvrtc := '';
      if ParamCount = 14 then Nvrtc := ParamStr(14);
      AddGemma3nVision(ParamStr(2), ParamStr(3), ParamStr(4), ParamStr(5), ParamStr(6),
        ParamStr(7), Nvrtc, Kit);
      WriteLn(UTF8Encode(ExpandFileName(ParamStr(5))));
    except
      on E: Exception do
      begin
        WriteLn(ErrOutput, UTF8Encode(E.Message));
        Halt(1);
      end;
    end;
    Halt(0);
  end;
  EffectiveCount := ParamCount;
  if (ParamCount >= 15) and (ParamStr(ParamCount - 2) = '--context') then Dec(EffectiveCount, 3);
  if not ((EffectiveCount = 12) or ((EffectiveCount = 13) and (ParamCount = 13) and
    (ParamStr(13) = '--text-onnx')) or ((EffectiveCount = 15) and (ParamStr(13) = '--cuda'))) then
    begin
      WriteLn(ErrOutput, 'Usage: ConvertGemma3n checkpoint.gguf projector.gguf-or-- recipe-dir destination');
      WriteLn(ErrOutput,
        '       ggml-base.dll sentencepiece.dll TVM-compiler-directory link.exe tvm_ffi.lib');
      WriteLn(ErrOutput,
        '       msvc-lib ucrt-lib windows-lib [--cuda nvrtc.dll sm_XX] [--context tokens prefill]');
      Halt(2);
    end;
  try
    Context := 0;
    Chunk := 0;
    if EffectiveCount <> ParamCount then
    begin
      Context := StrToInt(ParamStr(ParamCount - 1));
      Chunk := StrToInt(ParamStr(ParamCount));
    end;
    Projector := ParamStr(2);
    if Projector = '-' then Projector := '';
    ReadKit(7);
    if EffectiveCount = 15 then
      ConvertGemma3nCudaBundle(ParamStr(1), Projector, ParamStr(3), ParamStr(4), ParamStr(5), ParamStr(6),
        ParamStr(14), ParamStr(15), Kit, Context)
    else
      ConvertGemma3nCpuBundle(ParamStr(1), Projector, ParamStr(3), ParamStr(4), ParamStr(5),
        ParamStr(6), Kit, EffectiveCount <> 13, Context, Chunk);
    WriteLn(UTF8Encode(ExpandFileName(ParamStr(4))));
  except
    on E: Exception do
    begin
      WriteLn(ErrOutput, UTF8Encode(E.Message));
      Halt(1);
    end;
  end;
end.
