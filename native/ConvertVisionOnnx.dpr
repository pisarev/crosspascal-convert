{ ************************************************************************** }
{                                                                            }
{ ConvertVisionOnnx                                                          }
{                                                                            }
{ Copyright © 2026 Yuriy Pisarev                                             }
{                                                                            }
{ ************************************************************************** }

program ConvertVisionOnnx;

{$APPTYPE CONSOLE}
{$IFDEF FPC}{$MODE DELPHIUNICODE}{$ENDIF}

uses
  {$IFDEF FPC}SysUtils{$ELSE}System.SysUtils{$ENDIF}, CrossPascal.TVMCompiler, CrossPascal.VisionBundle,
  CrossPascal.NativeIO;

var
  Kit: TWindowsTvmKit;
  Recipe, Nvrtc, Architecture: string;
  RequiredEnd: Integer;

begin
  RequiredEnd := NativeParamCount;
  if (NativeParamCount in [13, 14]) and (NativeParamStr(NativeParamCount - 2) = '--cuda') then
  begin
    RequiredEnd := NativeParamCount - 3;
    Nvrtc := NativeParamStr(NativeParamCount - 1);
    Architecture := NativeParamStr(NativeParamCount);
  end;
  if not (RequiredEnd in [10, 11]) then
  begin
    WriteLn(ErrOutput, 'Usage: ConvertVisionOnnx checkpoint models.json profile-id destination');
    WriteLn(ErrOutput, '       TVM-compiler-directory link.exe tvm_ffi.lib msvc-lib ucrt-lib windows-lib [graph-recipe.onnx]');
    WriteLn(ErrOutput, '       [--cuda nvrtc.dll sm_89]');
    Halt(2);
  end;
  try
    Kit.CompilerDirectory := NativeParamStr(5);
    Kit.Linker := NativeParamStr(6);
    Kit.FfiLibrary := ExpandFileName(NativeParamStr(7));
    Kit.MsvcLibraries := NativeParamStr(8);
    Kit.UcrtLibraries := NativeParamStr(9);
    Kit.WindowsLibraries := NativeParamStr(10);
    Recipe := '';
    if RequiredEnd = 11 then Recipe := NativeParamStr(11);
    ConvertVisionBundle(NativeParamStr(1), NativeParamStr(2), NativeParamStr(3), NativeParamStr(4),
      Kit, Recipe, Nvrtc, Architecture);
    WriteLn(UTF8Encode(ExpandFileName(NativeParamStr(4))));
  except
    on E: Exception do
    begin
      WriteLn(ErrOutput, UTF8Encode(E.Message));
      Halt(1);
    end;
  end;
end.
