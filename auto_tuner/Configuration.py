import xml.sax
import ParseUtils
from Enums import *

####################################################################################################
class BaseConfiguration(xml.sax.ContentHandler):
    def loadFromDisk(self, filename):
        xml.sax.parse(filename, self)

####################################################################################################
class Pipeline(BaseConfiguration):
    def __init__(self):
        #self.globalSeed = -1
        self.deviceId = 0
        self.pgaBasePath = ""
        self.analyzerBin = ""
        self.partitionBin = ""
        self.binTemplatePath = ""
        self.workspaceTemplatePath = ""
        self.databaseScript = ""
        self.compileScript = ""
        self.compileDependenciesScript = ""
        self.cachePath = ""
        self.generatePartitions = False
        self.prepareWorkspaces = False
        self.compile = False
        self.findMaxNumAxioms = False
        self.run = False
        self.processResults = False
        self.saveToDatabase = False
        self.decorateDotFiles = False
        self.analyzerExecutionTimeout = 300
        self.dependenciesCompilationTimeout = 600
        self.partitionCompilationTimeout = 3600
        self.partitionExecutionTimeout = 600
        self.logLevel = LogLevel.DEBUG
        self.keepTemp = False
        self.rebuild = False
        self.compileDependencies = False
        self.useMultiprocess = False
        self.automaticPoolSize = False
        self.poolSize = 0
        self.cudaComputeCapability = None

    def startElement(self, name, attributes):
        if name == "Pipeline":
            #self.globalSeed = ParseUtils.try_to_parse_int(attributes.getValue("globalSeed"), -1)
            self.deviceId = ParseUtils.try_to_parse_int(attributes.getValue("deviceId"))
            self.pgaBasePath = attributes.getValue("pgaBasePath")
            self.analyzerBin = attributes.getValue("analyzerBin")
            self.partitionBin = attributes.getValue("partitionBin")
            self.binTemplatePath = attributes.getValue("binTemplatePath")
            self.workspaceTemplatePath = attributes.getValue("workspaceTemplatePath")
            self.databaseScript = attributes.getValue("databaseScript")
            self.compileScript = attributes.getValue("compileScript")
            self.compileDependenciesScript = attributes.getValue("compileDependenciesScript")
            self.cachePath = attributes.getValue("cachePath")
            self.generatePartitions = bool(ParseUtils.try_to_parse_int(attributes.getValue("generatePartitions")))
            self.prepareWorkspaces = bool(ParseUtils.try_to_parse_int(attributes.getValue("prepareWorkspaces")))
            self.compile = bool(ParseUtils.try_to_parse_int(attributes.getValue("compile")))
            self.findMaxNumAxioms = bool(ParseUtils.try_to_parse_int(attributes.getValue("findMaxNumAxioms")))
            self.run = bool(ParseUtils.try_to_parse_int(attributes.getValue("run")))
            self.processResults = bool(ParseUtils.try_to_parse_int(attributes.getValue("processResults")))
            self.saveToDatabase = bool(ParseUtils.try_to_parse_int(attributes.getValue("saveToDatabase")))
            self.decorateDotFiles = bool(ParseUtils.try_to_parse_int(attributes.getValue("decorateDotFiles")))
            self.analyzerExecutionTimeout = ParseUtils.try_to_parse_int(attributes.getValue("analyzerExecutionTimeout"))
            self.dependenciesCompilationTimeout = ParseUtils.try_to_parse_int(attributes.getValue("dependenciesCompilationTimeout"))
            self.partitionCompilationTimeout = ParseUtils.try_to_parse_int(attributes.getValue("partitionCompilationTimeout"))
            self.partitionExecutionTimeout = ParseUtils.try_to_parse_int(attributes.getValue("partitionExecutionTimeout"))
            # NOTE: if a valid log level is not found, the default value (LogLevel.DEBUG) is used
            if "logLevel" in attributes.keys():
                log_level_name = attributes.getValue("logLevel")
                for log_level, log_level_name_ in LogLevel.NAMES.items():
                    if log_level_name_ == log_level_name:
                        self.logLevel = log_level
            self.keepTemp = bool(ParseUtils.try_to_parse_int(attributes.getValue("keepTemp")))
            self.rebuild = bool(ParseUtils.try_to_parse_int(attributes.getValue("rebuild")))
            self.compileDependencies = bool(ParseUtils.try_to_parse_int(attributes.getValue("compileDependencies")))
            self.useMultiprocess = bool(ParseUtils.try_to_parse_int(attributes.getValue("useMultiprocess")))
            self.automaticPoolSize = bool(ParseUtils.try_to_parse_int(attributes.getValue("automaticPoolSize")))
            self.poolSize = ParseUtils.try_to_parse_int(attributes.getValue("poolSize"))
            self.cudaComputeCapability = attributes.getValue("cudaComputeCapability")

####################################################################################################
class Scene(BaseConfiguration):
    def __init__(self):
        self.runSeed = -1
        self.grammarFile = ""
        self.templateFile = ""
        self.outputPath = ""
        self.gridX = 0
        self.gridY = 0
        self.itemSize = 0
        self.maxNumVertices = 0
        self.maxNumIndices = 0
        self.maxNumElements = 0
        self.queuesMem = 0
        self.numRuns = 0
        self.minNumAxioms = 0
        self.maxNumAxioms = 0
        self.axiomStep = 0
        self.globalMinNumAxioms = -1
        self.optimization = 0
        self.instrumentation = False
        self.whippletreeTechnique = WhippletreeTechnique.MEGAKERNEL
        self.H1TO1 = False
        self.HRR = False
        self.HSTO = -1
        self.HTSPECS = ""
        self.RSAMPL = 0
        self.analyzerSeed = -1

    def startElement(self, name, attributes):
        if name == "Scene":
            if "runSeed" in attributes.keys():
                self.runSeed = ParseUtils.try_to_parse_int(attributes.getValue("runSeed"), -1)
            self.grammarFile = attributes.getValue("grammarFile")
            self.templateFile = attributes.getValue("templateFile")
            self.outputPath = attributes.getValue("outputPath")
            self.gridX = ParseUtils.try_to_parse_int(attributes.getValue("gridX"))
            self.gridY = ParseUtils.try_to_parse_int(attributes.getValue("gridY"))
            self.itemSize = ParseUtils.try_to_parse_int(attributes.getValue("itemSize"))
            if "maxNumVertices" in attributes.keys():
                self.maxNumVertices = ParseUtils.try_to_parse_int(attributes.getValue("maxNumVertices"))
            if "maxNumIndices" in attributes.keys():
                self.maxNumIndices = ParseUtils.try_to_parse_int(attributes.getValue("maxNumIndices"))
            if "maxNumElements" in attributes.keys():
                self.maxNumElements = ParseUtils.try_to_parse_int(attributes.getValue("maxNumElements"))
            self.queuesMem = ParseUtils.try_to_parse_int(attributes.getValue("queuesMem"))
            self.numRuns = ParseUtils.try_to_parse_int(attributes.getValue("numRuns"))
            self.minNumAxioms = ParseUtils.try_to_parse_int(attributes.getValue("minNumAxioms"))
            self.maxNumAxioms = ParseUtils.try_to_parse_int(attributes.getValue("maxNumAxioms"))
            self.axiomStep = ParseUtils.try_to_parse_int(attributes.getValue("axiomStep"))
            if "globalMinNumAxioms" in attributes.keys():
                self.globalMinNumAxioms = ParseUtils.try_to_parse_int(attributes.getValue("globalMinNumAxioms"))
            if "optimization" in attributes.keys():
                self.optimization = ParseUtils.try_to_parse_int(attributes.getValue("optimization"))
            if "instrumentation" in attributes.keys():
                self.instrumentation = ParseUtils.try_to_parse_bool(attributes.getValue("instrumentation"))
            # NOTE: if a valid whippletree technique is not found, the default value (WhippletreeTechnique.MEGAKERNELS) is used
            if "whippletreeTechnique" in attributes.keys():
                whippletree_technique_name = attributes.getValue("whippletreeTechnique")
                for whippletree_technique, whippletree_technique_name_ in WhippletreeTechnique.NAMES.items():
                    if whippletree_technique_name_ == whippletree_technique_name:
                        self.whippletreeTechnique = whippletree_technique
            if "H1TO1" in attributes.keys():
                self.H1TO1 = ParseUtils.try_to_parse_int(attributes.getValue("H1TO1"))
            if "HRR" in attributes.keys():
                self.HRR = ParseUtils.try_to_parse_int(attributes.getValue("HRR"))
            if "HSTO" in attributes.keys():
                self.HSTO = ParseUtils.try_to_parse_int(attributes.getValue("HSTO"))
            if "HTSPECS" in attributes.keys():
                self.HTSPECS = attributes.getValue("HTSPECS")
            if "RSAMPL" in attributes.keys():
                self.RSAMPL = ParseUtils.try_to_parse_int(attributes.getValue("RSAMPL"))
            if "analyzerSeed" in attributes.keys():
                self.analyzerSeed = ParseUtils.try_to_parse_int(attributes.getValue("analyzerSeed"))
