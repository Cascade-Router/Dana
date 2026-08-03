' Thin root wrapper - canonical script lives under scripts\launchers\
' Resolves scripts\launchers\stop_dana.vbs via ScriptFullName (VBS has no %~dp0).
Set fso = CreateObject("Scripting.FileSystemObject")
dir = fso.GetParentFolderName(WScript.ScriptFullName)
Set WshShell = CreateObject("WScript.Shell")
WshShell.Run "wscript.exe //B """ & dir & "\scripts\launchers\stop_dana.vbs""", 0, False
