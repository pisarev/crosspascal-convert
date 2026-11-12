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
  {$IFDEF FPC}SysUtils{$ELSE}System.SysUtils{$ENDIF}, CrossPascal.TVMCompiler, CrossPascal.VisionBundle;

var
  Kit: TWindowsTvmKit;
  Recipe, Nvrtc, Architecture: string;
  RequiredEnd: Integer;

begin
  RequiredEnd := ParamCount;
  if (ParamCount in [13, 14]) and (ParamStr(ParamCount - 2) = '--cuda') then
  begin
    RequiredEnd := ParamCount - 3;
    Nvrtc := ParamStr(ParamCount - 1);
    Architecture := ParamStr(ParamCount);
  end;
  if not (RequiredEnd in [10, 11]) then
  begin
    WriteLn(ErrOutput, 'Usage: ConvertVisionOnnx checkpoint models.json profile-id destination');
    WriteLn(ErrOutput, '       TVM-compiler-directory link.exe tvm_ffi.lib msvc-lib ucrt-lib windows-lib [graph-recipe.onnx]');
    WriteLn(ErrOutput, '       [--cuda nvrtc.dll sm_89]');
    Halt(2);
  end;
  try
    Kit.CompilerDirectory := ParamStr(5);
    Kit.Linker := ParamStr(6);
    Kit.FfiLibrary := ExpandFileName(ParamStr(7));
    Kit.MsvcLibraries := ParamStr(8);
    Kit.UcrtLibraries := ParamStr(9);
    Kit.WindowsLibraries := ParamStr(10);
    Recipe := '';
    if RequiredEnd = 11 then Recipe := ParamStr(11);
    ConvertVisionBundle(ParamStr(1), ParamStr(2), ParamStr(3), ParamStr(4), Kit, Recipe, Nvrtc, Architecture);
    WriteLn(UTF8Encode(ExpandFileName(ParamStr(4))));
  except
    on E: Exception do
    begin
      WriteLn(ErrOutput, UTF8Encode(E.Message));
      Halt(1);
    end;
  end;
end.
