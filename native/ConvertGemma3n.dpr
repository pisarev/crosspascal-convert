{ ************************************************************************** }
{                                                                            }
{ ConvertGemma3n                                                             }
{                                                                            }
{ Copyright © 2026 Yuriy Pisarev                                             }
{                                                                            }
{ ************************************************************************** }

program ConvertGemma3n;

{$APPTYPE CONSOLE}
{$IFDEF FPC}{$MODE DELPHIUNICODE}{$ENDIF}

uses
  {$IFDEF FPC}SysUtils{$ELSE}System.SysUtils{$ENDIF}, CrossPascal.TVMCompiler, CrossPascal.Gemma3nBundle,
  CrossPascal.NativeIO;

var
  Kit: TWindowsTvmKit;
  Projector, Nvrtc: string;
  EffectiveCount, Context, Chunk: Integer;

procedure ReadKit(const First: Integer);
begin
  Kit.CompilerDirectory := NativeParamStr(First);
  Kit.Linker := NativeParamStr(First + 1);
  Kit.FfiLibrary := ExpandFileName(NativeParamStr(First + 2));
  Kit.MsvcLibraries := NativeParamStr(First + 3);
  Kit.UcrtLibraries := NativeParamStr(First + 4);
  Kit.WindowsLibraries := NativeParamStr(First + 5);
end;

begin
  SetTextCodePage(Output, 65001);
  SetTextCodePage(ErrOutput, 65001);
  if NativeParamStr(1) = '--attach-vision' then
  begin
    if not (NativeParamCount in [13, 14]) then
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
      if NativeParamCount = 14 then Nvrtc := NativeParamStr(14);
      AddGemma3nVision(NativeParamStr(2), NativeParamStr(3), NativeParamStr(4), NativeParamStr(5),
        NativeParamStr(6), NativeParamStr(7), Nvrtc, Kit);
      WriteLn(UTF8Encode(ExpandFileName(NativeParamStr(5))));
    except
      on E: Exception do
      begin
        WriteLn(ErrOutput, UTF8Encode(E.Message));
        Halt(1);
      end;
    end;
    Halt(0);
  end;
  EffectiveCount := NativeParamCount;
  if (NativeParamCount >= 15) and (NativeParamStr(NativeParamCount - 2) = '--context') then
    Dec(EffectiveCount, 3);
  if not ((EffectiveCount = 12) or ((EffectiveCount = 13) and (NativeParamCount = 13) and
    (NativeParamStr(13) = '--text-onnx')) or ((EffectiveCount = 15) and (NativeParamStr(13) = '--cuda'))) then
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
    if EffectiveCount <> NativeParamCount then
    begin
      Context := StrToInt(NativeParamStr(NativeParamCount - 1));
      Chunk := StrToInt(NativeParamStr(NativeParamCount));
    end;
    Projector := NativeParamStr(2);
    if Projector = '-' then Projector := '';
    ReadKit(7);
    if EffectiveCount = 15 then
      ConvertGemma3nCudaBundle(NativeParamStr(1), Projector, NativeParamStr(3), NativeParamStr(4), NativeParamStr(5), NativeParamStr(6),
        NativeParamStr(14), NativeParamStr(15), Kit, Context)
    else
      ConvertGemma3nCpuBundle(NativeParamStr(1), Projector, NativeParamStr(3), NativeParamStr(4),
        NativeParamStr(5), NativeParamStr(6), Kit, EffectiveCount <> 13, Context, Chunk);
    WriteLn(UTF8Encode(ExpandFileName(NativeParamStr(4))));
  except
    on E: Exception do
    begin
      WriteLn(ErrOutput, UTF8Encode(E.Message));
      Halt(1);
    end;
  end;
end.
