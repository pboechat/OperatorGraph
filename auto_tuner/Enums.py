####################################################################################################
class LogLevel:
    DEBUG = 0
    INFO = 1
    WARNING = 2
    ERROR = 3
    FATAL = 4

    NAMES = {
        DEBUG: "DEBUG",
        INFO: "INFO",
        WARNING: "WARNING",
        ERROR: "ERROR",
        FATAL: "FATAL",
    }

####################################################################################################
class ReturnValue:
    SUCCESS = 0
    INVALID_PARAMETERS = 1
    GLEW_INITIALIZATION_ERROR = 2
    CREATE_DIRECTORY_ERROR = 3
    WINDOWS_API_ERROR = 4
    CUDA_EXCEPTION = 5
    GL_EXCEPTION = 6
    GEOMETRY_BUFFER_OVERFLOW_EXCEPTION = 7
    RUNTIME_ERROR = 8
    UNEXPECTED_ERROR = 9

    NAMES = {
        SUCCESS: "SUCCESS", 
        INVALID_PARAMETERS: "INVALID_PARAMETERS",
        GLEW_INITIALIZATION_ERROR: "GLEW_INITIALIZATION_ERROR",
        CREATE_DIRECTORY_ERROR: "CREATE_DIRECTORY_ERROR",
        WINDOWS_API_ERROR: "WINDOWS_API_ERROR", 
        CUDA_EXCEPTION: "CUDA_EXCEPTION", 
        GL_EXCEPTION: "GL_EXCEPTION", 
        GEOMETRY_BUFFER_OVERFLOW_EXCEPTION: "GEOMETRY_BUFFER_OVERFLOW_EXCEPTION", 
        RUNTIME_ERROR: "RUNTIME_ERROR", 
        UNEXPECTED_ERROR: "UNEXPECTED_ERROR"
    }

####################################################################################################
class WhippletreeTechnique:
    KERNELS = 0
    DYN_PAR = 1
    MEGAKERNEL = 2
    MEGAKERNEL_LOCAL_QUEUES = 3

    NAMES = {
        KERNELS: "KERNELS",
        DYN_PAR: "DYN_PAR",
        MEGAKERNEL: "MEGAKERNEL",
        MEGAKERNEL_LOCAL_QUEUES: "MEGAKERNEL_LOCAL_QUEUES"
    }