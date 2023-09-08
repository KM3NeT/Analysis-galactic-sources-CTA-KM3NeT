#!python3

""" Interface for configuration and help for executing scripts in workflow
"""

import yaml

# import nbformat
import pathlib

# from nbconvert.preprocessors import ExecutePreprocessor


class AnalysisConfig:
    """Class to manage file paths and configuration of the analysis"""

    def __init__(self, configpath=""):
        self.configfilepath = configpath
        self.config = {}
        self.statusfilepath = ".analysis_status.yml"
        self.datapath = ""

        rootdir = pathlib.Path(__file__).parents[0].resolve()
        self.read_analysis_config(rootdir / "analysis_standard_config.yml")
        self.read_analysis_config()
        self.set_datapath()

    def read_analysis_config(self, filename=""):
        """Reads the configuration"""
        thisconfig = {}
        if not self.configfilepath and not filename:
            rootdir = pathlib.Path(__file__).parents[1].resolve()
            self.configfilepath = rootdir / "Analysis/analysis_config.yml"
            try:
                with open(self.configfilepath, "r") as file:
                    thisconfig = yaml.safe_load(file)
            except:
                print(
                    "No configuration file in /Analysis found. Please provide analysis_config.yml!"
                )
        elif filename:
            if not str(filename).find("analysis_standard_config.yml") > 0:
                self.configfilepath = Path(filename)
            try:
                with open(filename, "r") as file:
                    thisconfig = yaml.safe_load(file)
            except:
                print(
                    "No configuration file",
                    filename,
                    "found. Please provide correct file path!",
                )
        else:
            try:
                with open(self.configfilepath, "r") as file:
                    thisconfig = yaml.safe_load(file)
            except:
                print(
                    "No configuration file ",
                    self.configfilepath,
                    " found. Please provide analysis_config.yml",
                )
        if thisconfig:
            for key in thisconfig:
                if key in self.config:
                    for k2 in thisconfig[key].keys():
                        self.config[key][k2] = thisconfig[key][k2]
                else:
                    self.config.setdefault(key, thisconfig[key])

    def set_datapath(self, datapath=""):
        """Retrieves path to data folder"""
        if not datapath:
            self.datapath = pathlib.Path(__file__).parents[1].resolve() / "data"
        elif get_value("datafolder", "io"):
            self.datapath = (
                self.configfilepath / get_value("datafolder", "io")
            ).resolve()
        else:
            self.datapath = pathlib.Path(datapath)

    def init_status(self, statusfile, writelocal=False):
        """Set initial parameters for statusfile"""
        self.read_analysis_config()
        self.config["status"] = {"current_source": self.get_value("sources")[0]}
        if writelocal:
            yaml.dump(firststatus, self.statusfile)

    def get_value(self, value, path=""):
        """Retrieve a configuration value, optionally provide path separated by ."""
        thisdic = self.config
        if path:
            for level in path.split("."):
                try:
                    thisdic = thisdic[level]
                except:
                    print("Did not find", level, "in config file path", path)
        try:
            return thisdic[value]
        except:
            print("Did not find", value, "in", thisdic)
            return ""

    def get_file(self, datapath):
        """Returns the full path for a given data file path"""
        return pathlib.PurePath(self.datapath, datapath)

    def get_source(self):
        """Return astro source name depending on execution status"""
        if self.get_value("execute") == "manual":
            return self.get_value("sources")[0]
        else:
            if self.get_value("current_source", "status"):
                return self.get_value("current_source", "status")
            else:
                return ""

    ### REWORK MARKER

    def get_status(self, reset=False):
        """Get status file keeping track of execution"""
        status = {}
        try:
            with open(statfilepath, "r") as statfile:
                status = yaml.safe_load(statfile)
                statfile.close()
                if reset:
                    init_status(open(statfilepath, "w"))
                    status = yaml.safe_load(open(statfilepath, "r"))
        except FileNotFoundError:
            with open(statfilepath, "w") as statfile:
                init_status(statfile)
            with open(statfilepath, "r") as statfile:
                status = yaml.safe_load(statfile)
        return status

    def change_status(self, key, value):
        """Change an entry in the status file"""
        status = get_status()
        if key in status:
            status[key] = value
        else:
            status.setdefault(key, value)
        with open(statfilepath, "w") as statfile:
            yaml.dump(status, statfile)


# def execute_notebook(nbfilename, outfilename = ""):
#     """Execute a notebook within a python script. If outfilename specified, the executed notebook is saved."""

#     with open(nbfilename) as f:
#         nb = nbformat.read(f, as_version=4)

#     ep = ExecutePreprocessor(kernel_name='km3net_cta')
#     ep.preprocess(nb)

#     if outfilename:
#         with open(outfilename, 'w', encoding='utf-8') as f:
#             nbformat.write(nb, f)


def iterate_sources():
    pass
