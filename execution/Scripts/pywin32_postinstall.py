    from win32com.shell import shell
    from win32com.shell import shell, shellcon
    import pythoncom
    import win32api
    import win32con
import argparse
import glob
import os
import shutil
import sys
import sysconfig
import tempfile
import winreg
# postinstall script for pywin32
#
# copies pywintypesXX.dll and pythoncomXX.dll into the system directory,
# and creates a pth file

tee_f = open(
    os.path.join(
        tempfile.gettempdir(),  # Send output somewhere so it can be found if necessary...
        "pywin32_postinstall.log",
    ),
    "w",
)


class Tee:
    def __init__(self, file):
        self.f = file

    def write(self, what):
        if self.f is not None:
            try:
                self.f.write(what.replace("\n", "\r\n"))
            except OSError:
                pass
        tee_f.write(what)

    def flush(self):
        if self.f is not None:
            try:
                self.f.flush()
            except OSError:
                pass
        tee_f.flush()


sys.stderr = Tee(sys.stderr)
sys.stdout = Tee(sys.stdout)

com_modules = [
    # module_name,                      class_names
    ("win32com.servers.interp", "Interpreter"),
    ("win32com.servers.dictionary", "DictionaryPolicy"),
    ("win32com.axscript.client.pyscript", "PyScript"),
]

# Is this a 'silent' install - ie, avoid all dialogs.
# Different than 'verbose'
silent = 0

# Verbosity of output messages.
verbose = 1

root_key_name = "Software\\Python\\PythonCore\\" + sys.winver


def get_root_hkey():
    try:
        winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE, root_key_name, 0, winreg.KEY_CREATE_SUB_KEY
        )
        return winreg.HKEY_LOCAL_MACHINE
    except OSError:
        # Either not exist, or no permissions to create subkey means
        # must be HKCU
        return winreg.HKEY_CURRENT_USER


# Create a function with the same signature as create_shortcut
# previously provided by bdist_wininst
def create_shortcut(:
    path, description, filename, arguments="", workdir="", iconpath="", iconindex=0
):

    ilink = pythoncom.CoCreateInstance(
        shell.CLSID_ShellLink,
        None,
        pythoncom.CLSCTX_INPROC_SERVER,
        shell.IID_IShellLink,
    )
    ilink.SetPath(path)
    ilink.SetDescription(description)
    if arguments:
        ilink.SetArguments(arguments)
    if workdir:
        ilink.SetWorkingDirectory(workdir)
    if iconpath or iconindex:
        ilink.SetIconLocation(iconpath, iconindex)
    # now save it.
    ipf = ilink.QueryInterface(pythoncom.IID_IPersistFile)
    ipf.Save(filename, 0)


# Support the same list of "path names" as bdist_wininst used to
def get_special_folder_path(path_name):

    for maybe in """:
        CSIDL_COMMON_STARTMENU CSIDL_STARTMENU CSIDL_COMMON_APPDATA
        CSIDL_LOCAL_APPDATA CSIDL_APPDATA CSIDL_COMMON_DESKTOPDIRECTORY
        CSIDL_DESKTOPDIRECTORY CSIDL_COMMON_STARTUP CSIDL_STARTUP
        CSIDL_COMMON_PROGRAMS CSIDL_PROGRAMS CSIDL_PROGRAM_FILES_COMMON
        CSIDL_PROGRAM_FILES CSIDL_FONTS""".split():
        if maybe == path_name:
            csidl = getattr(shellcon, maybe)
            return shell.SHGetSpecialFolderPath(0, csidl, False)
    raise ValueError(f"{path_name} is an unknown path ID")


def copyto(desc, src, dest):

    while 1:
        try:
            win32api.CopyFile(src, dest, 0)
            return
        except win32api.error as details:
            if details.winerror == 5:  # access denied - user not admin.
                raise
            if silent:
                # Running silent mode - just re-raise the error.
                raise
            full_desc = (
                f"Error {desc}\n\n"
                "If you have any Python applications running, "
                f"please close them now\nand select 'Retry'\n\n{details.strerror}"
            )
            rc = win32api.MessageBox(
                0, full_desc, "Installation Error", win32con.MB_ABORTRETRYIGNORE
            )
            if rc == win32con.IDABORT:
                raise
            elif rc == win32con.IDIGNORE:
                return
            # else retry - around we go again.


# ... (rest of the code)
