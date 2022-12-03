from csnake import CodeWriter, Variable, Function  # ,FormattedLiteral, TextModifier
import json
import sys
import os
import pyopencl as cl
# import numpy as np

'''
platforms = cl.get_platforms()
ret_num_platforms = len(platforms)
platform_id = hex(platforms[0].int_ptr)
devices = platforms[0].get_devices(device_type=cl.device_type.ALL)
ret_num_devices = len(devices)
device_id = hex(devices[0].int_ptr)
max_work_group_size = devices[0].get_info(cl.device_info.MAX_WORK_GROUP_SIZE)

print(platforms)
print(ret_num_platforms)
print(platform_id)
print(devices)
print(ret_num_devices)
print(device_id)
print(max_work_group_size)
'''

numPlatforms = len(cl.get_platforms())
nameJson = "data.json"
nameC = "BNN.c"
if __name__ == "__main__":
    argLength = len(sys.argv)
    if argLength < 2:
        print("No argument passed. Using data.json to create C-code...\n")
    elif argLength == (2 or 3):
        nameJson = str(sys.argv[1])
        if nameJson.endswith(".json"):
            print("Using " + nameJson + " to create C-code...\n")
        else:
            nameJson = "data.json"
            print("File passed isn't a JSON-file. Using data.json to create C-code...\n")
        if argLength == 3:
            nameC = str(sys.argv[2])
            if not nameC.endswith(".c"):
                nameC = nameC + ".c"
    else:
        print("Too many arguments passed. Using data.json to create C-code...\n")

cw = CodeWriter()

try:
    jsonSize = os.path.getsize(nameJson)
    with open(nameJson) as f:
        data = json.load(f)

    height = 0
    width = 0
    channels = 0
    kernelHeight = 0
    kernelWidth = 0
    kernelChannels = 0
    if ("height" in data and "width" in data and "channels" in data and
            "kernelHeight" in data and "kernelWidth" in data and "kernelChannels" in data):
        isCNN = True
        print("Passed BNN is a CNN.\n")
        height = data["height"]
        width = data["width"]
        channels = data["channels"]
        kernelHeight = data["kernelHeight"]
        kernelWidth = data["kernelWidth"]
        kernelChannels = data["kernelChannels"]
    else:
        isCNN = False

    numBiases = len(data["biases"])
    numWeights = len(data["weights"])
    numBatchBiases = len(data["batchBiases"])
    numScales = len(data["scales"])
    dataKeys = numBiases + numWeights + numBatchBiases + numScales
    if (numBiases != numWeights) | (numBatchBiases != numScales) | (numBatchBiases != (numBiases - 1)):
        print("Too many/few keys in JSON")

    numLayers = int(numBiases)
    if isCNN:
        numOutputs = int(numLayers + 3)
    else:
        numOutputs = int(numLayers + 1)

    # Reading size of outputs
    outputSize = [0 for i in range(int(numOutputs))]
    HSize = [0 for i in range(int(numOutputs - 2))]
    WSize = [0 for i in range(int(numOutputs - 2))]
    if isCNN:
        outputSize[0] = int(height * width)
        HSize[0] = int(height)
        WSize[0] = int(width)
        for i in range(int(numOutputs - 3)):
            if i % 2 == 0:
                HSize[i + 1] = HSize[i] - (kernelHeight - 1)
                WSize[i + 1] = WSize[i] - (kernelWidth - 1)
            else:
                HSize[i + 1] = int(HSize[i] / 2)
                WSize[i + 1] = int(WSize[i] / 2)
            outputSize[i + 1] = int(HSize[i + 1] * WSize[i + 1] * channels)
        outputSize[numOutputs - 2] = channels
        outputSize[numOutputs - 1] = len(data["biases"][numBiases - 1])
    else:
        outputSize[0] = int(len(data["weights"][0]) / len(data["biases"][0]))
        for i in range(int(numLayers)):
            outputSize[i + 1] = len(data["biases"][i])

    cw.include("<stdio.h>")
    cw.include("<stdlib.h>")
    cw.include("<math.h>")
    cw.include("<time.h>")
    cw.include("<CL/cl.h>")
    cw.include("<json.h>")
    cw.include("<string.h>")
    cw.include("<pthread.h>")
    cw.add_line(" ")
    cw.add_define("HAVE_STRUCT_TIMESPEC")
    cw.add_define("JSON_SIZE " + str(jsonSize))
    cw.add_line(" ")

    cw.add_line("// Struct for arguments")
    cw.add_line("struct bnn_struct")
    cw.add_line("{")
    cw.add_line("    int* output0;")
    cw.add_line("    char* nameOutput;")
    cw.add_line("    char* jsonOutput;")
    cw.add_line("    int currentInput;")
    cw.add_line("    int numInputs;")
    cw.add_line("    char parallel;")
    if isCNN:
        cw.add_line("    char* source_str1;")
        cw.add_line("    char* source_str2;")
        cw.add_line("    char* source_str3;")
        cw.add_line("    size_t source_size1;")
        cw.add_line("    size_t source_size2;")
        cw.add_line("    size_t source_size3;")
    else:
        cw.add_line("    char* source_str;")
        cw.add_line("    size_t source_size;")
    for i in range(int(numLayers)):
        cw.add_line("    double* bias" + str(i + 1) + ";")
        cw.add_line("    double* weight" + str(i + 1) + ";")
        if i != int(numLayers - 1):
            cw.add_line("    double* batchBias" + str(i + 1) + ";")
            cw.add_line("    double* scale" + str(i + 1) + ";")
    cw.add_line("};")
    cw.add_line(" ")

    cw.add_line("// Size for outputs")
    if isCNN:
        for i in range(int(numOutputs)):
            cw.add_line("".join(("const int OUTPUT", str(i), " = ", str(outputSize[i]), ";")))
        cw.add_line("".join(("int* out", str(numOutputs - 3), "ptr = &OUTPUT", str(numOutputs - 3), ";")))
        cw.add_line("".join(("int* out", str(numOutputs - 2), "ptr = &OUTPUT", str(numOutputs - 2), ";")))
        cw.add_line(" ")

        cw.add_line("// CNN data")
        cw.add_line("const int CHANNELS = " + str(channels) + ";")
        for i in range(int(numWeights)):
            cw.add_line("const int WEIGHT" + str(i + 1) + " = " + str(len(data["weights"][i])) + ";")
        cw.add_line(" ")

        cw.add_line("const int sizeM = " + str(kernelChannels) + ";")
        cw.add_line("const int sizeKH = " + str(kernelHeight) + ";")
        cw.add_line("const int sizeKW = " + str(kernelWidth) + ";")
        cw.add_line("const int sizeC0 = 1;")
        cw.add_line("const int sizeC = " + str(channels) + ";")
        cw.add_line("int* sizeMptr = &sizeM;")
        cw.add_line("int* sizeKHptr = &sizeKH;")
        cw.add_line("int* sizeKWptr = &sizeKW;")
        cw.add_line("int* sizeC0ptr = &sizeC0;")
        cw.add_line("int* sizeCptr = &sizeC;")
        cw.add_line(" ")

        for i in range(int(numOutputs - 2)):
            cw.add_line("const int sizeH" + str(i) + " = " + str(HSize[i]) + ";")
            cw.add_line("const int sizeW" + str(i) + " = " + str(WSize[i]) + ";")
        for i in range(int(numOutputs - 2)):
            cw.add_line("int* sizeH" + str(i) + "ptr = &sizeH" + str(i) + ";")
            cw.add_line("int* sizeW" + str(i) + "ptr = &sizeW" + str(i) + ";")
        cw.add_line(" ")
    else:
        sizesForOutput = [" " for i in range(int(numOutputs * 2 + 1))]
        for i in range(int(numOutputs)):
            sizesForOutput[i] = "".join(("const int OUTPUT", str(i), " = ", str(outputSize[i]), ";"))
            sizesForOutput[i + int(numOutputs)] = "".join(("int* out", str(i), "ptr = &OUTPUT", str(i), ";"))
        cw.add_lines(sizesForOutput)

    batchNormParameters = [
        "// BatchNorm parameters",
        "const char bnT = 1;",
        "const char bnF = 0;",
        "char* bnTrue = &bnT;",
        "char* bnFalse = &bnF;",
        " "
    ]
    cw.add_lines(batchNormParameters)

    arg1_setLocalSize = Variable("global_item_size", "size_t")
    arg2_setLocalSize = Variable("maxWorkSize", "size_t")
    setLocalSize = Function(
        "setLocalSize",
        "size_t",
        arguments=(arg1_setLocalSize, arg2_setLocalSize)
    )
    setLocalSize.add_code((
        "if (maxWorkSize <= 0)",
        "{",
        "    return 1;",
        "}",
        "if (global_item_size <= maxWorkSize)",
        "{",
        "    return global_item_size;",
        "}",
        "else",
        "{",
        "    size_t temp_Size = global_item_size;",
        "    for (int i = 2; temp_Size > maxWorkSize; i++)",
        "    {",
        "        if (global_item_size % i == 0)",
        "        {",
        "            temp_Size = global_item_size / i;",
        "        }",
        "    }",
        "    return temp_Size;",
        "}"
    ))
    cw.add_function_definition(setLocalSize)
    cw.add_line(" ")

    cw.add_line("void *calcBNN(void* arguments)")
    cw.add_line("{")
    cw.add_line("    // Read the arguments")
    cw.add_line("    struct bnn_struct* args = arguments;")
    cw.add_line(" ")

    cw.add_line("    // Clock for time measurement")
    cw.add_line("    clock_t begin;")
    cw.add_line("    clock_t end;")
    timeSpent = "    double "
    for i in range(int(numOutputs)):
        timeSpent = "".join((timeSpent, "time_spent", str(i + 1)))
        if i != int(numOutputs - 1):
            timeSpent = "".join((timeSpent, ", "))
        else:
            timeSpent = "".join((timeSpent, ";"))
    cw.add_line(timeSpent)
    cw.add_line(" ")

    globalLocalSize = [
        "    // Size for parallel processing in OpenCL",
        "    size_t global_item_size = 0;",
        "    // CL_DEVICE_MAX_WORK_GROUP_SIZE",
        "    size_t local_item_size = 0;",
        " "
    ]
    cw.add_lines(globalLocalSize)

    createFile = [
        "    // Creating file for outputs",
        r'    FILE* outputText = fopen(args -> nameOutput, "a+");',
        "    if (outputText == NULL)",
        "    {",
        r'        printf("Error opening the file %s", args -> nameOutput);',
        "        return -1;",
        "    }",
        r'    fprintf(outputText, "%d. input calculation:\n", args -> currentInput);',
        r'    fprintf(outputText, "Input:\n[");',
        r'    if ((args -> parallel) == 1)',
        r'    {',
        r'        printf("%d. input calculation:\n", args -> currentInput);',
        r'        printf("Input:\n[");',
        r'    }',
        "    for (int i = 0; i < OUTPUT0; i++)",
        "    {",
        r'        fprintf(outputText, "%d", args -> output0[i]);',
        r'        if ((args -> parallel) == 1)',
        r'        {',
        r'            printf("%d", args -> output0[i]);',
        r'        }',
        "        if (i != (OUTPUT0 - 1))",
        "        {",
        r'            fprintf(outputText, ", ");',
        r'            if ((args -> parallel) == 1)',
        r'            {',
        r'                printf(", ");',
        r'            }',
        "        }",
        "        else",
        "        {",
        r'            fprintf(outputText, "]");',
        r'            if ((args -> parallel) == 1)',
        r'            {',
        r'                printf("]");',
        r'            }',
        "        }",
        "    }",
        r'    fprintf(outputText, "\n");',
        r'    if ((args -> parallel) == 1)',
        r'    {',
        r'        printf("\n");',
        r'    }',
        " "
    ]
    cw.add_lines(createFile)

    # Load CL-code

    deviceInfo = [
        "    // Get platform and device information",
        "    cl_platform_id* platform_id = (cl_platform_id*)malloc(" + str(numPlatforms) +
        " * sizeof(cl_platform_id));",
        "    cl_device_id device_id = NULL;",
        "    cl_uint ret_num_devices;",
        "    cl_uint ret_num_platforms;",
        "    cl_int ret = clGetPlatformIDs(" + str(numPlatforms) + ", platform_id, &ret_num_platforms);",
        " ",
        "    ret = clGetDeviceIDs(platform_id[0], CL_DEVICE_TYPE_ALL, 1, &device_id, &ret_num_devices);",
        " ",
        "    // Get maximum size for work group",
        "    size_t maxWorkSize;",
        "    size_t usedWorkSize;",
        "    ret = clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &maxWorkSize, NULL);",
        " "
    ]
    cw.add_lines(deviceInfo)

    cw.add_line("    if ((args -> parallel) == 0)")
    cw.add_line("    {")
    cw.add_line("        usedWorkSize = maxWorkSize / (args -> numInputs);")
    cw.add_line("    }")
    cw.add_line("    else")
    cw.add_line("    {")
    cw.add_line("        usedWorkSize = maxWorkSize;")
    cw.add_line("    }")
    cw.add_line(" ")

    contextCommandQueue = [
        "    // Create an OpenCL context",
        "    cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);",
        " ",
        "    // Create a command queue",
        "    cl_command_queue command_queue = clCreateCommandQueue(context, device_id, 0, &ret);",
        " "
    ]
    cw.add_lines(contextCommandQueue)

    batchNormBuffers = [
        "    // Create batchNorm buffers",
        "    cl_mem batchnormTrue_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(char), NULL, &ret);",
        "    ret = clEnqueueWriteBuffer(command_queue, batchnormTrue_buffer, CL_TRUE, 0, sizeof(char), "
        "bnTrue, 0, NULL, NULL);",
        "    cl_mem batchnormFalse_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(char), NULL, &ret);",
        "    ret = clEnqueueWriteBuffer(command_queue, batchnormFalse_buffer, CL_TRUE, 0, sizeof(char), "
        "bnFalse, 0, NULL, NULL);",
        " "
    ]
    cw.add_lines(batchNormBuffers)

    clearMemory = [" " for i in range(int(numLayers * 6 - 3))]
    if isCNN:
        cw.add_line("    // Create CNN buffers")
        cw.add_line("    cl_mem M_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int), NULL, &ret);")
        cw.add_line("    ret = clEnqueueWriteBuffer(command_queue, M_buffer, CL_TRUE, 0, sizeof(int), sizeMptr, 0,"
                    " NULL, NULL);")
        cw.add_line("    cl_mem KH_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int), NULL, &ret);")
        cw.add_line("    ret = clEnqueueWriteBuffer(command_queue, KH_buffer, CL_TRUE, 0, sizeof(int), sizeKHptr,"
                    " 0, NULL, NULL);")
        cw.add_line("    cl_mem KW_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int), NULL, &ret);")
        cw.add_line("    ret = clEnqueueWriteBuffer(command_queue, KW_buffer, CL_TRUE, 0, sizeof(int), sizeKWptr,"
                    " 0, NULL, NULL);")
        cw.add_line("    cl_mem C0_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int), NULL, &ret);")
        cw.add_line("    ret = clEnqueueWriteBuffer(command_queue, C0_buffer, CL_TRUE, 0, sizeof(int), sizeC0ptr,"
                    " 0, NULL, NULL);")
        cw.add_line("    cl_mem C_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int), NULL, &ret);")
        cw.add_line("    ret = clEnqueueWriteBuffer(command_queue, C_buffer, CL_TRUE, 0, sizeof(int), sizeCptr,"
                    " 0, NULL, NULL);")
        cw.add_line("")

        cw.add_line("    // Create output0 buffers")
        cw.add_line("    cl_mem output0_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, OUTPUT0 * sizeof(int),"
                    " NULL, &ret);")
        cw.add_line("    ret = clEnqueueWriteBuffer(command_queue, output0_buffer, CL_TRUE, 0, OUTPUT0 * sizeof(int),"
                    " args -> output0, 0, NULL, NULL);")
        cw.add_line(" ")

        for i in range(int(numLayers)):
            cw.add_line("    // Create layer" + str(i + 1) + " buffers")
            if i < int(numLayers - 2):
                cw.add_line(
                    "".join(
                        ("    cl_mem output", str(i * 2 + 1),
                         "_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, OUTPUT",
                         str(i * 2 + 1), " * sizeof(int), NULL, &ret);"))
                )
                cw.add_line(
                    "".join(
                        ("    cl_mem output", str(i * 2 + 2),
                         "_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, OUTPUT",
                         str(i * 2 + 2), " * sizeof(int), NULL, &ret);"))
                )
                if i == 0:
                    cw.add_line("    cl_mem H0_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY,"
                                " sizeof(int), NULL, &ret);")
                    cw.add_line("    ret = clEnqueueWriteBuffer(command_queue, H0_buffer, CL_TRUE,"
                                " 0, sizeof(int), sizeH0ptr, 0, NULL, NULL);")
                    cw.add_line("    cl_mem W0_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY,"
                                " sizeof(int), NULL, &ret);")
                    cw.add_line("    ret = clEnqueueWriteBuffer(command_queue, W0_buffer, CL_TRUE,"
                                " 0, sizeof(int), sizeW0ptr, 0, NULL, NULL);")
                cw.add_line(
                    "".join(
                        ("    cl_mem H", str(i * 2 + 1),
                         "_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int), NULL, &ret);"))
                )
                cw.add_line(
                    "".join(
                        ("    ret = clEnqueueWriteBuffer(command_queue, H",
                         str(i * 2 + 1), "_buffer, CL_TRUE, 0, sizeof(int), sizeH",
                         str(i * 2 + 1), "ptr, 0, NULL, NULL);"))
                )
                cw.add_line(
                    "".join(
                        ("    cl_mem W", str(i * 2 + 1),
                         "_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int), NULL, &ret);"))
                )
                cw.add_line(
                    "".join(
                        ("    ret = clEnqueueWriteBuffer(command_queue, W",
                         str(i * 2 + 1), "_buffer, CL_TRUE, 0, sizeof(int), sizeW",
                         str(i * 2 + 1), "ptr, 0, NULL, NULL);"))
                )
                cw.add_line(
                    "".join(
                        ("    cl_mem H", str(i * 2 + 2),
                         "_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int), NULL, &ret);"))
                )
                cw.add_line(
                    "".join(
                        ("    ret = clEnqueueWriteBuffer(command_queue, H",
                         str(i * 2 + 2), "_buffer, CL_TRUE, 0, sizeof(int), sizeH",
                         str(i * 2 + 2), "ptr, 0, NULL, NULL);"))
                )
                cw.add_line(
                    "".join(
                        ("    cl_mem W", str(i * 2 + 2),
                         "_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int), NULL, &ret);"))
                )
                cw.add_line(
                    "".join(
                        ("    ret = clEnqueueWriteBuffer(command_queue, W",
                         str(i * 2 + 2), "_buffer, CL_TRUE, 0, sizeof(int), sizeW",
                         str(i * 2 + 2), "ptr, 0, NULL, NULL);"))
                )
            else:
                cw.add_line(
                    "".join(
                        ("    cl_mem output", str(i + 3),
                         "_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, OUTPUT",
                         str(i + 3), " * sizeof(int), NULL, &ret);"))
                )
                cw.add_line(
                    "".join(
                        ("    cl_mem output", str(i + 2),
                         "_size_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int), NULL, &ret);"))
                )
                cw.add_line(
                    "".join(
                        ("    ret = clEnqueueWriteBuffer(command_queue, output", str(i + 2),
                         "_size_buffer, CL_TRUE, 0, sizeof(int), out", str(i + 2),
                         "ptr, 0, NULL, NULL);"))
                )
            if i != int(numLayers - 1):
                cw.add_line(
                    "".join(
                        ("    cl_mem bias", str(i + 1),
                         "_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY,"
                         " CHANNELS * sizeof(double), NULL, &ret);"))
                )
                cw.add_line(
                    "".join(
                        ("    ret = clEnqueueWriteBuffer(command_queue, bias",
                         str(i + 1), "_buffer, CL_TRUE, 0, CHANNELS * sizeof(double), args -> bias",
                         str(i + 1), ", 0, NULL, NULL);"))
                )
            else:
                cw.add_line(
                    "".join(
                        ("    cl_mem bias", str(i + 1),
                         "_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, OUTPUT", str(i + 3),
                         " * sizeof(double), NULL, &ret);"))
                )
                cw.add_line(
                    "".join(
                        ("    ret = clEnqueueWriteBuffer(command_queue, bias",
                         str(i + 1), "_buffer, CL_TRUE, 0, OUTPUT", str(i + 3), " * sizeof(double), args -> bias",
                         str(i + 1), ", 0, NULL, NULL);"))
                )
            cw.add_line(
                "".join(("    cl_mem weight", str(i + 1),
                         "_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, (WEIGHT", str(i + 1),
                         ") * sizeof(double), NULL, &ret);"))
            )
            cw.add_line(
                "".join(("    ret = clEnqueueWriteBuffer(command_queue, weight", str(i + 1),
                         "_buffer, CL_TRUE, 0, (WEIGHT", str(i + 1), ") * sizeof(double), args -> weight",
                         str(i + 1), ", 0, NULL, NULL);"))
            )
            if i != int(numLayers - 1):
                cw.add_line(
                    "".join(("    cl_mem batchbias", str(i + 1),
                             "_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY,"
                             " CHANNELS * sizeof(double), NULL, &ret);"))
                )
                cw.add_line(
                    "".join(("    ret = clEnqueueWriteBuffer(command_queue, batchbias", str(i + 1),
                             "_buffer, CL_TRUE, 0, CHANNELS * sizeof(double), args -> batchBias", str(i + 1),
                             ", 0, NULL, NULL);"))
                )
                cw.add_line(
                    "".join(
                        ("    cl_mem scale", str(i + 1),
                         "_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY,"
                         " CHANNELS * sizeof(double), NULL, &ret);"))
                )
                cw.add_line(
                    "".join(("    ret = clEnqueueWriteBuffer(command_queue, scale", str(i + 1),
                             "_buffer, CL_TRUE, 0, CHANNELS * sizeof(double), args -> scale", str(i + 1),
                             ", 0, NULL, NULL);"))
                )
            cw.add_line(" ")
    else:
        output0Buffers = [
            "    // Create output0 buffers",
            "    cl_mem output0_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, OUTPUT0 * sizeof(int), NULL, &ret);",
            "    cl_mem output0_size_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int), NULL, &ret);",
            "    ret = clEnqueueWriteBuffer(command_queue, output0_buffer, CL_TRUE, 0, OUTPUT0 * sizeof(int),"
            " args -> output0, 0, NULL, NULL);",
            "    ret = clEnqueueWriteBuffer(command_queue, output0_size_buffer, CL_TRUE, 0, sizeof(int),"
            " out0ptr, 0, NULL, NULL);",
            " "
        ]
        cw.add_lines(output0Buffers)

        for i in range(int(numLayers)):
            cw.add_line("    // Create layer" + str(i + 1) + " buffers")
            cw.add_line(
                "".join(("    cl_mem output", str(i + 1), "_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, OUTPUT",
                         str(i + 1), " * sizeof(int), NULL, &ret);"))
            )
            cw.add_line(
                "".join(("    cl_mem output", str(i + 1), "_size_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, "
                                                          "sizeof(int), NULL, &ret);"))
            )
            cw.add_line(
                "".join(
                    ("    ret = clEnqueueWriteBuffer(command_queue, output", str(i + 1), "_size_buffer, CL_TRUE, 0, "
                                                                                         "sizeof(int), out", str(i + 1),
                     "ptr, 0, NULL, NULL);"))
            )
            cw.add_line(
                "".join(("    cl_mem bias", str(i + 1), "_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, OUTPUT",
                         str(i + 1), " * sizeof(double), NULL, &ret);"))
            )
            cw.add_line(
                "".join(
                    ("    ret = clEnqueueWriteBuffer(command_queue, bias", str(i + 1), "_buffer, CL_TRUE, 0, OUTPUT",
                     str(i + 1), " * sizeof(double), args -> bias", str(i + 1), ", 0, NULL, NULL);"))
            )
            cw.add_line(
                "".join(("    cl_mem weight", str(i + 1), "_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, (OUTPUT",
                         str(i + 1), " * OUTPUT", str(i), ") * sizeof(double), NULL, &ret);"))
            )
            cw.add_line(
                "".join(
                    ("    ret = clEnqueueWriteBuffer(command_queue, weight", str(i + 1), "_buffer, CL_TRUE, 0, (OUTPUT",
                     str(i + 1), " * OUTPUT", str(i), ") * sizeof(double), args -> weight",
                     str(i + 1), ", 0, NULL, NULL);"))
            )

            if i != int(numLayers - 1):
                cw.add_line(
                    "".join(("    cl_mem batchbias", str(i + 1),
                             "_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, OUTPUT",
                             str(i + 1), " * sizeof(double), NULL, &ret);"))
                )
                cw.add_line(
                    "".join(("    ret = clEnqueueWriteBuffer(command_queue, batchbias", str(i + 1),
                             "_buffer, CL_TRUE, 0, OUTPUT", str(i + 1), " * sizeof(double), args -> batchBias",
                             str(i + 1), ", 0, NULL, NULL);"))
                )
                cw.add_line(
                    "".join(
                        ("    cl_mem scale", str(i + 1), "_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, OUTPUT",
                         str(i + 1), " * sizeof(double), NULL, &ret);"))
                )
                cw.add_line(
                    "".join(("    ret = clEnqueueWriteBuffer(command_queue, scale", str(i + 1),
                             "_buffer, CL_TRUE, 0, OUTPUT", str(i + 1), " * sizeof(double), args -> scale",
                             str(i + 1), ", 0, NULL, NULL);"))
                )

            if i != int(numLayers - 1):
                clearMemory[6 * i] = "    ret = clReleaseMemObject(output" + str(i + 1) + "_buffer);"
                clearMemory[6 * i + 1] = "    ret = clReleaseMemObject(output" + str(i + 1) + "_size_buffer);"
                clearMemory[6 * i + 2] = "    ret = clReleaseMemObject(bias" + str(i + 1) + "_buffer);"
                clearMemory[6 * i + 3] = "    ret = clReleaseMemObject(weight" + str(i + 1) + "_buffer);"
                clearMemory[6 * i + 4] = "    ret = clReleaseMemObject(batchbias" + str(i + 1) + "_buffer);"
                clearMemory[6 * i + 5] = "    ret = clReleaseMemObject(scale" + str(i + 1) + "_buffer);"
            else:
                clearMemory[6 * i] = "    ret = clReleaseMemObject(output" + str(i + 1) + "_buffer);"
                clearMemory[6 * i + 1] = "    ret = clReleaseMemObject(bias" + str(i + 1) + "_buffer);"
                clearMemory[6 * i + 2] = "    ret = clReleaseMemObject(weight" + str(i + 1) + "_buffer);"
            cw.add_line(" ")

    if isCNN:
        cw.add_line("    // Create programs from the kernel source")
        cw.add_line("    cl_program program1 = clCreateProgramWithSource(context, 1,"
                    " (const char**)&args -> source_str1, (const size_t*)&args -> source_size1, &ret);")
        cw.add_line("    cl_program program2 = clCreateProgramWithSource(context, 1,"
                    " (const char**)&args -> source_str2, (const size_t*)&args -> source_size2, &ret);")
        cw.add_line("    cl_program program3 = clCreateProgramWithSource(context, 1,"
                    " (const char**)&args -> source_str3, (const size_t*)&args -> source_size3, &ret);")
        cw.add_line(" ")

        cw.add_line("    // Build the programs")
        cw.add_line("    ret = clBuildProgram(program1, 1, &device_id, NULL, NULL, NULL);")
        cw.add_line("    ret = clBuildProgram(program2, 1, &device_id, NULL, NULL, NULL);")
        cw.add_line("    ret = clBuildProgram(program3, 1, &device_id, NULL, NULL, NULL);")
        cw.add_line(" ")

        cw.add_line("    // Show errors if programs couldn't be built")
        cw.add_line("    if (ret == CL_BUILD_PROGRAM_FAILURE)")
        cw.add_line("    {")
        cw.add_line("        // Determine the size of the logs")
        cw.add_line("        size_t log_size1;")
        cw.add_line("        size_t log_size2;")
        cw.add_line("        size_t log_size3;")
        cw.add_line("        clGetProgramBuildInfo(program1, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size1);")
        cw.add_line("        clGetProgramBuildInfo(program2, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size2);")
        cw.add_line("        clGetProgramBuildInfo(program3, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size3);")
        cw.add_line(" ")

        cw.add_line("        // Allocate memory for the logs")
        cw.add_line("        char* log1 = (char*)malloc(log_size1);")
        cw.add_line("        char* log2 = (char*)malloc(log_size2);")
        cw.add_line("        char* log3 = (char*)malloc(log_size3);")
        cw.add_line(" ")

        cw.add_line("        // Get the logs")
        cw.add_line("        clGetProgramBuildInfo(program1, device_id, CL_PROGRAM_BUILD_LOG, log_size1, log1, NULL);")
        cw.add_line("        clGetProgramBuildInfo(program2, device_id, CL_PROGRAM_BUILD_LOG, log_size2, log2, NULL);")
        cw.add_line("        clGetProgramBuildInfo(program3, device_id, CL_PROGRAM_BUILD_LOG, log_size3, log3, NULL);")
        cw.add_line(" ")

        cw.add_line("        // Print the logs")
        cw.add_line(r'        printf("log1:\n%s\n\nlog2:\n%s\n\nlog3:\n%s\n", log1, log2, log3);')
        cw.add_line(" ")

        cw.add_line(r'        // Free the logs')
        cw.add_line(r'        free(log1);')
        cw.add_line(r'        free(log2);')
        cw.add_line(r'        free(log3);')
        cw.add_line("    }")
        cw.add_line(" ")

        cw.add_line("    // Create the OpenCL kernel")
        cw.add_line(r'    cl_kernel kernel1 = clCreateKernel(program1, "cnn_1", &ret);')
        cw.add_line(r'    cl_kernel kernel2 = clCreateKernel(program2, "cnn_2", &ret);')
        cw.add_line(r'    cl_kernel kernel3 = clCreateKernel(program3, "cnn_3", &ret);')
    else:
        programKernel = [
            "    // Create a program from the kernel source",
            "    cl_program program = clCreateProgramWithSource(context, 1, (const char**)&args -> source_str, "
            "(const size_t*)&args -> source_size, &ret);",
            " ",
            "    // Build the program",
            "    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);",
            " ",
            "     // Show errors if program couldn't be built",
            "    if (ret == CL_BUILD_PROGRAM_FAILURE)",
            "    {",
            "        // Determine the size of the log",
            "        size_t log_size;",
            "        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);",
            " ",
            "        // Allocate memory for the log",
            "        char* log = (char*)malloc(log_size);",
            " ",
            "        // Get the log",
            "        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);",
            " ",
            "        // Print the log",
            r'        printf("%s\n", log);',
            r'       // Free the log',
            r'        free(log);',
            "    }",
            " ",
            "    // Create the OpenCL kernel",
            r'    cl_kernel kernel = clCreateKernel(program, "bnn_1", &ret);'
        ]
        cw.add_lines(programKernel)
    cw.add_line(" ")

    # Setting parameters and perform computation in OpenCL

    if isCNN:
        for i in range(int(numLayers)):
            if i < int(numLayers - 2):
                cw.add_line("    // Set the arguments of kernel1 to get output" + str(i * 2 + 1))
                cw.add_line("    ret = clSetKernelArg(kernel1, 0, sizeof(cl_mem), (void*)&W" +
                            str(i * 2) + "_buffer);")
                cw.add_line("    ret = clSetKernelArg(kernel1, 1, sizeof(cl_mem), (void*)&W" +
                            str(i * 2 + 1) + "_buffer);")
                if i == 0:
                    cw.add_line("    ret = clSetKernelArg(kernel1, 2, sizeof(cl_mem), (void*)&M_buffer);")
                    cw.add_line("    ret = clSetKernelArg(kernel1, 3, sizeof(cl_mem), (void*)&KH_buffer);")
                    cw.add_line("    ret = clSetKernelArg(kernel1, 4, sizeof(cl_mem), (void*)&KW_buffer);")
                    cw.add_line("    ret = clSetKernelArg(kernel1, 5, sizeof(cl_mem), (void*)&C0_buffer);")
                if i == 1:
                    cw.add_line("    ret = clSetKernelArg(kernel1, 5, sizeof(cl_mem), (void*)&C_buffer);")
                cw.add_line("    ret = clSetKernelArg(kernel1, 6, sizeof(cl_mem), (void*)&output" +
                            str(i * 2) + "_buffer);")
                cw.add_line("    ret = clSetKernelArg(kernel1, 7, sizeof(cl_mem), (void*)&output" +
                            str(i * 2 + 1) + "_buffer);")
                cw.add_line(
                    "    ret = clSetKernelArg(kernel1, 8, sizeof(cl_mem), (void*)&bias" +
                    str(i + 1) + "_buffer);")
                cw.add_line(
                    "    ret = clSetKernelArg(kernel1, 9, sizeof(cl_mem), (void*)&weight" +
                    str(i + 1) + "_buffer);")
                cw.add_line(
                    "    ret = clSetKernelArg(kernel1, 10, sizeof(cl_mem), (void*)&batchbias" +
                    str(i + 1) + "_buffer);")
                cw.add_line(
                    "    ret = clSetKernelArg(kernel1, 11, sizeof(cl_mem), (void*)&scale" +
                    str(i + 1) + "_buffer);")
                cw.add_line(" ")

                cw.add_line("    // Execute kernel1 to get output" + str(i * 2 + 1))
                cw.add_line("    global_item_size = sizeH" + str(i * 2 + 1) + " * sizeW" + str(i * 2 + 1) + ";")
                cw.add_line("    local_item_size = setLocalSize(global_item_size, usedWorkSize);")
                cw.add_line(" ")

                cw.add_line("    begin = clock();")
                cw.add_line(
                    "    ret = clEnqueueNDRangeKernel(command_queue, kernel1, 1, NULL, &global_item_size,"
                    " &local_item_size, 0, NULL, NULL);"
                )
                cw.add_line("    end = clock();")
                cw.add_line("    time_spent" + str(i * 2 + 1) + " = (double)(end - begin) / CLOCKS_PER_SEC;")
                cw.add_line(
                    r'    fprintf(outputText, "Output' + str(i * 2 + 1) +
                    r' calculation time: %f seconds\n", time_spent' + str(i * 2 + 1) + r');'
                )
                cw.add_line("    if ((args -> parallel) == 1)")
                cw.add_line("    {")
                cw.add_line(
                    r'        printf("Output' + str(i * 2 + 1) + r' calculation time: %f seconds\n", time_spent' +
                    str(i * 2 + 1) + r');'
                )
                cw.add_line("    }")
                cw.add_line(" ")

                cw.add_line("    // Set the arguments of kernel2 to get output" + str(i * 2 + 2))
                cw.add_line("    ret = clSetKernelArg(kernel2, 0, sizeof(cl_mem), (void*)&W" +
                            str(i * 2 + 1) + "_buffer);")
                cw.add_line("    ret = clSetKernelArg(kernel2, 1, sizeof(cl_mem), (void*)&W" +
                            str(i * 2 + 2) + "_buffer);")
                if i == 0:
                    cw.add_line("    ret = clSetKernelArg(kernel2, 2, sizeof(cl_mem), (void*)&C_buffer);")
                cw.add_line("    ret = clSetKernelArg(kernel2, 3, sizeof(cl_mem), (void*)&output" +
                            str(i * 2 + 1) + "_buffer);")
                cw.add_line("    ret = clSetKernelArg(kernel2, 4, sizeof(cl_mem), (void*)&output" +
                            str(i * 2 + 2) + "_buffer);")
                cw.add_line(" ")

                cw.add_line("    // Execute kernel2 to get output" + str(i * 2 + 2))
                cw.add_line("    global_item_size = sizeH" + str(i * 2 + 2) + " * sizeW" + str(i * 2 + 2) + ";")
                cw.add_line("    local_item_size = setLocalSize(global_item_size, usedWorkSize);")
                cw.add_line(" ")

                cw.add_line("    begin = clock();")
                cw.add_line(
                    "    ret = clEnqueueNDRangeKernel(command_queue, kernel2, 1, NULL, &global_item_size,"
                    " &local_item_size, 0, NULL, NULL);"
                )
                cw.add_line("    end = clock();")
                cw.add_line("    time_spent" + str(i * 2 + 2) + " = (double)(end - begin) / CLOCKS_PER_SEC;")
                cw.add_line(
                    r'    fprintf(outputText, "Output' + str(i * 2 + 2) +
                    r' calculation time: %f seconds\n", time_spent' + str(i * 2 + 2) + r');'
                )
                cw.add_line("    if ((args -> parallel) == 1)")
                cw.add_line("    {")
                cw.add_line(
                    r'        printf("Output' + str(i * 2 + 2) + r' calculation time: %f seconds\n", time_spent' +
                    str(i * 2 + 2) + r');'
                )
                cw.add_line("    }")
                cw.add_line(" ")
            else:
                cw.add_line("    // Set the arguments of kernel3 to get output" + str(i + 3))
                cw.add_line("    ret = clSetKernelArg(kernel3, 0, sizeof(cl_mem), (void*)&output" +
                            str(i + 2) + "_buffer);")
                cw.add_line(
                    "    ret = clSetKernelArg(kernel3, 1, sizeof(cl_mem), (void*)&output" + str(i + 2) +
                    "_size_buffer);")
                cw.add_line(
                    "    ret = clSetKernelArg(kernel3, 2, sizeof(cl_mem), (void*)&output" + str(i + 3) +
                    "_buffer);")
                if i == int(numLayers - 2):
                    cw.add_line("    ret = clSetKernelArg(kernel3, 3, sizeof(cl_mem), (void*)&batchnormTrue_buffer);")
                if i == int(numLayers - 1):
                    cw.add_line("    ret = clSetKernelArg(kernel3, 3, sizeof(cl_mem), (void*)&batchnormFalse_buffer);")
                cw.add_line(
                    "    ret = clSetKernelArg(kernel3, 4, sizeof(cl_mem), (void*)&bias" + str(i + 1) + "_buffer);")
                cw.add_line(
                    "    ret = clSetKernelArg(kernel3, 5, sizeof(cl_mem), (void*)&weight" + str(i + 1) + "_buffer);")
                if i != int(numLayers - 1):
                    cw.add_line(
                        "    ret = clSetKernelArg(kernel3, 6, sizeof(cl_mem), (void*)&batchbias" +
                        str(i + 1) + "_buffer);")
                    cw.add_line(
                        "    ret = clSetKernelArg(kernel3, 7, sizeof(cl_mem), (void*)&scale" +
                        str(i + 1) + "_buffer);")
                cw.add_line(" ")

                cw.add_line("    // Execute kernel3 to get output" + str(i + 3))
                cw.add_line("    global_item_size = OUTPUT" + str(i + 3) + ";")
                cw.add_line("    local_item_size = setLocalSize(global_item_size, usedWorkSize);")
                cw.add_line(" ")

                cw.add_line("    begin = clock();")
                cw.add_line(
                    "    ret = clEnqueueNDRangeKernel(command_queue, kernel3, 1, NULL, &global_item_size,"
                    " &local_item_size, 0, NULL, NULL);"
                )
                cw.add_line("    end = clock();")
                cw.add_line("    time_spent" + str(i + 3) + " = (double)(end - begin) / CLOCKS_PER_SEC;")
                cw.add_line(
                    r'    fprintf(outputText, "Output' + str(i + 3) + r' calculation time: %f seconds\n", time_spent' +
                    str(i + 3) + r');'
                )
                cw.add_line("    if ((args -> parallel) == 1)")
                cw.add_line("    {")
                cw.add_line(
                    r'        printf("Output' + str(i + 3) + r' calculation time: %f seconds\n", time_spent' +
                    str(i + 3) + r');'
                )
                cw.add_line("    }")
                cw.add_line(" ")
    else:
        for i in range(int(numLayers)):
            cw.add_line("    // Set the arguments of the kernel to get output" + str(i + 1))
            cw.add_line("    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&output" + str(i) + "_buffer);")
            cw.add_line(
                "    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&output" + str(i) + "_size_buffer);")
            cw.add_line("    ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&output" + str(i + 1) + "_buffer);")
            if i == 0:
                cw.add_line("    ret = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&batchnormTrue_buffer);")
            if i == int(numLayers - 1):
                cw.add_line("    ret = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&batchnormFalse_buffer);")
            cw.add_line("    ret = clSetKernelArg(kernel, 4, sizeof(cl_mem), (void*)&bias" + str(i + 1) + "_buffer);")
            cw.add_line("    ret = clSetKernelArg(kernel, 5, sizeof(cl_mem), (void*)&weight" + str(i + 1) + "_buffer);")
            if i != int(numLayers - 1):
                cw.add_line(
                    "    ret = clSetKernelArg(kernel, 6, sizeof(cl_mem), (void*)&batchbias" + str(i + 1) + "_buffer);"
                )
                cw.add_line(
                    "    ret = clSetKernelArg(kernel, 7, sizeof(cl_mem), (void*)&scale" + str(i + 1) + "_buffer);")
            cw.add_line(" ")

            cw.add_line("    // Execute the OpenCL kernel to get output" + str(i + 1))
            cw.add_line("    global_item_size = OUTPUT" + str(i + 1) + ";")
            cw.add_line("    local_item_size = setLocalSize(global_item_size, usedWorkSize);")
            cw.add_line(" ")

            cw.add_line("    begin = clock();")
            cw.add_line(
                "    ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_item_size,"
                " &local_item_size, 0, NULL, NULL);"
            )
            cw.add_line("    end = clock();")
            cw.add_line("    time_spent" + str(i + 1) + " = (double)(end - begin) / CLOCKS_PER_SEC;")
            cw.add_line(
                r'    fprintf(outputText, "Output' + str(i + 1) + r' calculation time: %f seconds\n", time_spent' +
                str(i + 1) + r');'
            )
            cw.add_line("    if ((args -> parallel) == 1)")
            cw.add_line("    {")
            cw.add_line(
                r'    printf("Output' + str(i + 1) + r' calculation time: %f seconds\n", time_spent' +
                str(i + 1) + r');'
            )
            cw.add_line("    }")
            cw.add_line(" ")

    # Read last output and run logSoftmax

    timeSpent = [" " for i in range(int(numOutputs))]
    for i in range(int(numOutputs)):
        if i != int(numOutputs - 1):
            timeSpent[i] = "time_spent" + str(i + 1) + " + "
        else:
            timeSpent[i] = "time_spent" + str(i + 1)

    if isCNN:
        lastOutputNum = numOutputs - 1
    else:
        lastOutputNum = numLayers

    cw.add_line(
        "    // Read the memory buffer output" + str(lastOutputNum) + " on the device to the local variable output" +
        str(lastOutputNum)
    )
    cw.add_line("    int* output" + str(lastOutputNum) + " = (int*)malloc(sizeof(int) * OUTPUT" +
                str(lastOutputNum) + ");")
    cw.add_line(
        "    ret = clEnqueueReadBuffer(command_queue, output" + str(lastOutputNum) + "_buffer, CL_TRUE, "
        "0, OUTPUT" + str(lastOutputNum) + " * sizeof(int), output" + str(lastOutputNum) + ", 0, NULL, NULL);"
    )
    cw.add_line(" ")

    cw.add_line("    // Malloc output" + str(numOutputs))
    cw.add_line(
        "    double* output" + str(numOutputs) + " = (double*)malloc(sizeof(double) * OUTPUT" +
        str(lastOutputNum) + ");"
    )
    cw.add_line("    for (int i = 0; i < OUTPUT" + str(lastOutputNum) + "; i++)")
    cw.add_line("    {")
    cw.add_line("        output" + str(numOutputs) + "[i] = 0;")
    cw.add_line("    }")
    cw.add_line(" ")

    cw.add_line("    // Run LogSoftmax on output" + str(lastOutputNum))
    cw.add_line("    double max = 0;")
    cw.add_line("    double sum = 0;")
    cw.add_line("    double outLogSoftmax[" + str(outputSize[numOutputs - 1]) + "];")
    cw.add_line(" ")

    cw.add_line("    begin = clock();")
    cw.add_line("    for (int i = 0; i < OUTPUT" + str(lastOutputNum) + "; i++)")
    cw.add_line("    {")
    cw.add_line(
        "        max = (double)output" + str(lastOutputNum) + "[i] >= max ? (double)output" +
        str(lastOutputNum) + "[i] : max;")
    cw.add_line("    }")
    cw.add_line("    for (int i = 0; i < OUTPUT" + str(lastOutputNum) + "; i++)")
    cw.add_line("    {")
    cw.add_line("        outLogSoftmax[i] = exp((double)(output" + str(lastOutputNum) + "[i] - max));")
    cw.add_line("        sum += outLogSoftmax[i];")
    cw.add_line("    }")
    cw.add_line("    for (int i = 0; i < OUTPUT" + str(lastOutputNum) + "; i++)")
    cw.add_line("    {")
    cw.add_line("        outLogSoftmax[i] = log((double)(outLogSoftmax[i] / sum));")
    cw.add_line("    }")
    cw.add_line("    for (int i = 0; i < OUTPUT" + str(lastOutputNum) + "; i++)")
    cw.add_line("    {")
    cw.add_line("        output" + str(numOutputs) + "[i] += outLogSoftmax[i];")
    cw.add_line("    }")
    cw.add_line("    end = clock();")
    cw.add_line("    time_spent" + str(numOutputs) + " = (double)(end - begin) / CLOCKS_PER_SEC;")
    cw.add_line(
        r'    fprintf(outputText, "LogSoftmax calculation time: %f seconds\n", time_spent' +
        str(numOutputs) + r');'
    )
    cw.add_line(
        r'    fprintf(outputText, "Entire calculation time: %f seconds\n", ' +
        "".join(str(i) for i in timeSpent) + r');'
    )
    cw.add_line("    if ((args -> parallel) == 1)")
    cw.add_line("    {")
    cw.add_line(
        r'        printf("LogSoftmax calculation time: %f seconds\n", time_spent' +
        str(numOutputs) + r');'
    )
    cw.add_line(
        r'        printf("Entire calculation time: %f seconds\n", ' +
        "".join(str(i) for i in timeSpent) + r');'
    )
    cw.add_line("    }")
    cw.add_line(" ")

    # Open and write outputs in json

    cw.add_line("    // Open and write outputs in json")
    cw.add_line(r'    FILE* jsonText = fopen(args -> jsonOutput, "a+");')
    cw.add_line("    if (jsonText == NULL)")
    cw.add_line("    {")
    cw.add_line(r'        printf("Error opening the file %s", args -> jsonOutput);')
    cw.add_line("        return -1;")
    cw.add_line("    }")
    cw.add_line(" ")

    # Display the result to the screen

    cw.add_line("    // Display the result to the screen")
    cw.add_line("    double maxNum = -1.0;")
    cw.add_line("    double outPrint;")
    cw.add_line(" ")

    cw.add_line(r'    fprintf(outputText, "Output:\n[");')
    cw.add_line(r'    fprintf(jsonText, "    [ ");')
    cw.add_line("    if ((args -> parallel) == 1)")
    cw.add_line("    {")
    cw.add_line(r'        printf("Output:\n[");')
    cw.add_line("    }")
    cw.add_line("    for (int i = 0; i < OUTPUT" + str(lastOutputNum) + "; i++)")
    cw.add_line("    {")
    cw.add_line("        outPrint = (exp(output" + str(numOutputs) + "[i]));")
    cw.add_line("        if (outPrint > maxNum)")
    cw.add_line("        {")
    cw.add_line("            maxNum = outPrint;")
    cw.add_line("        }")
    cw.add_line(r'        fprintf(outputText, "%d: %f", i, outPrint);')
    cw.add_line(r'        fprintf(jsonText, "%f", outPrint);')
    cw.add_line("        if ((args -> parallel) == 1)")
    cw.add_line("        {")
    cw.add_line(r'            printf("%d: %f", i, outPrint);')
    cw.add_line("        }")
    cw.add_line("        if (i != (OUTPUT" + str(lastOutputNum) + " - 1))")
    cw.add_line("        {")
    cw.add_line(r'            fprintf(outputText, ", ");')
    cw.add_line(r'            fprintf(jsonText, ", ");')
    cw.add_line("            if ((args -> parallel) == 1)")
    cw.add_line("            {")
    cw.add_line(r'                printf(", ");')
    cw.add_line("            }")
    cw.add_line("        }")
    cw.add_line("        else")
    cw.add_line("        {")
    cw.add_line(r'            fprintf(outputText, "]");')
    cw.add_line(r'            fprintf(jsonText, " ]");')
    cw.add_line("            if ((args -> parallel) == 1)")
    cw.add_line("            {")
    cw.add_line(r'                printf("]");')
    cw.add_line("            }")
    cw.add_line("        }")
    cw.add_line("    }")
    cw.add_line(r'    fprintf(jsonText, "\n");')
    cw.add_line(r'    fprintf(jsonText, "  ]\n");')
    cw.add_line(r'    fprintf(jsonText, "}");')
    cw.add_line(r' ')

    cw.add_line(r'    char first = 1;')
    cw.add_line(r'    fprintf(outputText, "\nThe most likely output is the");')
    cw.add_line(r'    if ((args -> parallel) == 1)')
    cw.add_line(r'    {')
    cw.add_line(r'        printf("\nThe most likely output is the");')
    cw.add_line(r'    }')
    cw.add_line(r'    for (int i = 0; i < OUTPUT' + str(lastOutputNum) + r'; i++)')
    cw.add_line(r'    {')
    cw.add_line(r'        if (exp(output' + str(numOutputs) + r'[i]) == maxNum)')
    cw.add_line(r'        {')
    cw.add_line(r'            if (first == 1)')
    cw.add_line(r'            {')
    cw.add_line(r'                fprintf(outputText, " %d. output", i);')
    cw.add_line(r'                if ((args -> parallel) == 1)')
    cw.add_line(r'                {')
    cw.add_line(r'                    printf(" %d. output", i);')
    cw.add_line(r'                }')
    cw.add_line(r'                first = 0;')
    cw.add_line(r'            }')
    cw.add_line(r'            else')
    cw.add_line(r'            {')
    cw.add_line(r'                fprintf(outputText, " and the %d. output", i);')
    cw.add_line(r'                if ((args -> parallel) == 1)')
    cw.add_line(r'                {')
    cw.add_line(r'                    printf(" and the %d. output", i);')
    cw.add_line(r'                }')
    cw.add_line(r'            }')
    cw.add_line(r'        }')
    cw.add_line(r'    }')
    cw.add_line(r'    fprintf(outputText, " with %f%%.\n\n", maxNum * 100);')
    cw.add_line(r'    if ((args -> parallel) == 1)')
    cw.add_line(r'    {')
    cw.add_line(r'        printf(" with %f%%.\n\n", maxNum * 100);')
    cw.add_line(r'    }')
    cw.add_line(r' ')

    cw.add_line("    // Clean up")
    cw.add_line("    fclose(outputText);")
    cw.add_line("    fclose(jsonText);")
    cw.add_line(" ")

    cw.add_line("    ret = clFlush(command_queue);")
    cw.add_line("    ret = clFinish(command_queue);")
    if isCNN:
        cw.add_line("    ret = clReleaseKernel(kernel1);")
        cw.add_line("    ret = clReleaseKernel(kernel2);")
        cw.add_line("    ret = clReleaseKernel(kernel3);")
        cw.add_line("    ret = clReleaseProgram(program1);")
        cw.add_line("    ret = clReleaseProgram(program2);")
        cw.add_line("    ret = clReleaseProgram(program3);")
    else:
        cw.add_line("    ret = clReleaseKernel(kernel);")
        cw.add_line("    ret = clReleaseProgram(program);")
    cw.add_line("    ret = clReleaseCommandQueue(command_queue);")
    cw.add_line("    ret = clReleaseContext(context);")
    cw.add_line("    ret = clReleaseMemObject(batchnormTrue_buffer);")
    cw.add_line("    ret = clReleaseMemObject(batchnormFalse_buffer);")
    cw.add_line(" ")

    if isCNN:
        cw.add_line("    ret = clReleaseMemObject(M_buffer);")
        cw.add_line("    ret = clReleaseMemObject(KH_buffer);")
        cw.add_line("    ret = clReleaseMemObject(KW_buffer);")
        cw.add_line("    ret = clReleaseMemObject(C0_buffer);")
        cw.add_line("    ret = clReleaseMemObject(C_buffer);")
        for i in range(int(numOutputs)):
            cw.add_line("    ret = clReleaseMemObject(output" + str(i) + "_buffer);")
        cw.add_line("    ret = clReleaseMemObject(output" + str(numOutputs - 3) + "_size_buffer);")
        cw.add_line("    ret = clReleaseMemObject(output" + str(numOutputs - 2) + "_size_buffer);")
        for i in range(int(numLayers + 1)):
            cw.add_line("    ret = clReleaseMemObject(H" + str(i) + "_buffer);")
            cw.add_line("    ret = clReleaseMemObject(W" + str(i) + "_buffer);")
        for i in range(int(numLayers)):
            cw.add_line("    ret = clReleaseMemObject(bias" + str(i + 1) + "_buffer);")
            cw.add_line("    ret = clReleaseMemObject(weight" + str(i + 1) + "_buffer);")
            if i != int(numLayers - 1):
                cw.add_line("    ret = clReleaseMemObject(batchbias" + str(i + 1) + "_buffer);")
                cw.add_line("    ret = clReleaseMemObject(scale" + str(i + 1) + "_buffer);")
    else:
        cw.add_line("    ret = clReleaseMemObject(output0_size_buffer);")
        cw.add_line("    ret = clReleaseMemObject(output0_buffer);")
        cw.add_lines(clearMemory)
    cw.add_line(" ")

    if isCNN:
        cw.add_line("    free(output" + str(numOutputs - 1) + ");")
        cw.add_line("    free(output" + str(numOutputs) + ");")
    else:
        cw.add_line("    free(output" + str(numLayers) + ");")
        cw.add_line("    free(output" + str(numLayers + 1) + ");")

    cw.add_line("}")
    cw.add_line(" ")

    cw.add_line("int main(int argc, char **argv)")
    cw.add_line("{")
    cw.add_line(r'    // Json object for the inputs')
    cw.add_line(r'    json_object* json_inputs;')
    cw.add_line(r'    json_object* parsed_inputs;')
    cw.add_line(r' ')

    cw.add_line("    // Read arguments")
    cw.add_line(r'    char* nameInput = "inputs.json";')
    cw.add_line(r'    char parallel = 0;')
    cw.add_line("    if (argc < 2)")
    cw.add_line("    {")
    cw.add_line(r'        printf("No argument passed. Using inputs.json as input for BNN...\n\n");')
    cw.add_line("    }")
    cw.add_line(r'    else if (argc == 2)')
    cw.add_line(r'    {')
    cw.add_line(r'        if (strcmp(argv[1], "-p") == 0)')
    cw.add_line(r'        {')
    cw.add_line(r'            parallel = 1;')
    cw.add_line(r'            printf("Using parallelized internal BNN processing. '
                r'Using inputs.json as input for BNN...\n\n");')
    cw.add_line(r'        }')
    cw.add_line(r'        else')
    cw.add_line(r'        {')
    cw.add_line(r'            nameInput = argv[1];')
    cw.add_line(r'            printf("Using %s as input for BNN...\n\n", nameInput);')
    cw.add_line(r'        }')
    cw.add_line(r'    }')
    cw.add_line(r'    else if (argc == 3)')
    cw.add_line(r'    {')
    cw.add_line(r'        if (strcmp(argv[1], "-p") == 0)')
    cw.add_line(r'        {')
    cw.add_line(r'            parallel = 1;')
    cw.add_line(r'            nameInput = argv[2];')
    cw.add_line(r'            printf("Using parallelized internal BNN processing. Using %s as '
                r'input for BNN...\n\n", nameInput);')
    cw.add_line(r'        }')
    cw.add_line(r'        else if (strcmp(argv[2], "-p") == 0)')
    cw.add_line(r'        {')
    cw.add_line(r'            parallel = 1;')
    cw.add_line(r'            nameInput = argv[1];')
    cw.add_line(r'            printf("Using parallelized internal BNN processing. Using %s as input '
                r'for BNN...\n\n", nameInput);')
    cw.add_line(r'        }')
    cw.add_line(r'        else')
    cw.add_line(r'        {')
    cw.add_line(r'            printf("Wrong arguments passed. Use \"-p\" to parallelize internal BNN processing. '
                r'Using inputs.json as input for BNN...\n\n");')
    cw.add_line(r'        }')
    cw.add_line(r'    }')
    cw.add_line(r'    else')
    cw.add_line(r'    {')
    cw.add_line(r'        printf("Too many arguments passed. Using inputs.json as input for BNN...\n\n");')
    cw.add_line(r'    }')
    cw.add_line(r' ')

    cw.add_line("    // Read the inputs for the BNN")
    cw.add_line("    FILE* inputsBNN;")
    cw.add_line(r'    inputsBNN = fopen(nameInput, "r");')
    cw.add_line("    if (!inputsBNN)")
    cw.add_line("    {")
    cw.add_line(r'        fprintf(stderr, "Failed to load inputs for the BNN.\n");')
    cw.add_line("        exit(1);")
    cw.add_line("    }")
    cw.add_line(r'    fseek(inputsBNN, 0, SEEK_END);')
    cw.add_line(r'    size_t inputsSize = ftell(inputsBNN);')
    cw.add_line(r'    fseek(inputsBNN, 0, SEEK_SET);')
    cw.add_line(r'    char* inputsBuffer = (char*)malloc(inputsSize * sizeof(char));')
    cw.add_line("    fread(inputsBuffer, inputsSize, 1, inputsBNN);")
    cw.add_line("    fclose(inputsBNN);")
    cw.add_line("    parsed_inputs = json_tokener_parse(inputsBuffer);")
    cw.add_line(r'    json_object_object_get_ex(parsed_inputs, "inputs", &json_inputs);')
    cw.add_line(" ")

    cw.add_line("    // Read number of inputs")
    cw.add_line("    int numInputs = json_object_array_length(json_inputs);")
    cw.add_line(" ")

    cw.add_line("    // Create arguments")
    cw.add_line("    struct bnn_struct* args = (struct bnn_struct*)malloc(sizeof(struct bnn_struct) * numInputs);")
    cw.add_line(" ")

    cw.add_line("    // Name for output-file")
    cw.add_line("    char nameOutput[30];")
    cw.add_line("    char jsonOutput[30];")
    cw.add_line("    int numTime;")
    cw.add_line(r'    FILE* jsonText;')
    cw.add_line(r'    for (int i = 0; i < numInputs; i++)')
    cw.add_line(r'    {')
    cw.add_line(r'        numTime = (int)time(NULL);')
    cw.add_line(r'        snprintf(nameOutput, 30, "%s_%d_%d%s", "output", i, numTime, ".txt");')
    cw.add_line(r'        snprintf(jsonOutput, 30, "%s_%d_%d%s", "output", i, numTime, ".json");')
    cw.add_line(r'        jsonText = fopen(jsonOutput, "a+");')
    cw.add_line(r'        if (jsonText == NULL)')
    cw.add_line(r'        {')
    cw.add_line(r'            printf("Error opening the file %s", jsonOutput);')
    cw.add_line(r'            return -1;')
    cw.add_line(r'        }')
    cw.add_line(r'        fprintf(jsonText, "{\n");')
    cw.add_line(r'        fprintf(jsonText, "  \"outputs\":\n");')
    cw.add_line(r'        fprintf(jsonText, "  [\n");')
    cw.add_line(r'        fclose(jsonText);')
    cw.add_line(r'        args[i].nameOutput = (char*)malloc(sizeof(char) * 30);')
    cw.add_line(r'        args[i].jsonOutput = (char*)malloc(sizeof(char) * 30);')
    cw.add_line(r'        strcpy(args[i].nameOutput, nameOutput);')
    cw.add_line(r'        strcpy(args[i].jsonOutput, jsonOutput);')
    cw.add_line(r'        args[i].currentInput = i;')
    cw.add_line(r'        args[i].numInputs = numInputs;')
    cw.add_line(r'        if (parallel == 1)')
    cw.add_line(r'        {')
    cw.add_line(r'            args[i].parallel = 1;')
    cw.add_line(r'        }')
    cw.add_line(r'        else')
    cw.add_line(r'        {')
    cw.add_line(r'            args[i].parallel = 0;')
    cw.add_line(r'        }')
    cw.add_line(r'    }')
    cw.add_line(r' ')

    if isCNN:
        cw.add_line("    // Load the kernel source codes into the source_str arrays")
        cw.add_line(r'    char* source_str1 = (char*)malloc(' + str(1582 + len(str(channels))) + r' * sizeof(char));')
        cw.add_line(r'    source_str1 =')
        cw.add_line(r'        "__kernel void cnn_1(__global int* W0, __global int* W1, __global int* M,'
                    r' __global int* KH, __global int* KW, __global int* C, __global int* output0,'
                    r' __global int* output1, __global double* bias,'
                    r' __global double* weight, __global double* batchBias, __global double* scale)\n"')
        cw.add_line(r'        "{\n"')
        cw.add_line(r'        "    // Get the index of the current element\n"')
        cw.add_line(r'        "    int gid = get_global_id(0);\n"')
        cw.add_line(r'        "    \n"')
        cw.add_line(r'        "    // Local variables\n"')
        cw.add_line(r'        "    double localOut[' + str(channels) + r']; // Size of channels\n"')
        cw.add_line(r'        "    int outNum = M[0] * gid;\n"')
        cw.add_line(r'        "    int calcHelp = outNum;\n"')
        cw.add_line(r'        "    int calcH = outNum / (W1[0] * M[0]);\n"')
        cw.add_line(r'        "    calcHelp -= (calcH * W1[0] * M[0]);\n"')
        cw.add_line(r'        "    int calcW = calcHelp / M[0];\n"')
        cw.add_line(r'        "    \n"')
        cw.add_line(r'        "    // Conv\n"')
        cw.add_line(r'        "    for (int m = 0; m < M[0]; m++) {\n"')
        cw.add_line(r'        "        localOut[m] = bias[m];\n"')
        cw.add_line(r'        "    }\n"')
        cw.add_line(r'        "    for (int kH = 0; kH < KH[0]; kH++) {\n"')
        cw.add_line(r'        "        for (int kW = 0; kW < KW[0]; kW++) {\n"')
        cw.add_line(r'        "            for (int c = 0; c < C[0]; c++) {\n"')
        cw.add_line(r'        "                for (int m = 0; m < M[0]; m++) {\n"')
        cw.add_line(r'        "                    localOut[m] += weight[m + c * M[0] + kW * (M[0] * C[0]) +'
                    r' kH * (M[0] * C[0] * KW[0])] * output0[c + (calcW + kW) * (C[0]) + (calcH + kH) *'
                    r' (C[0] * W0[0])];\n"')
        cw.add_line(r'        "                }\n"')
        cw.add_line(r'        "            }\n"')
        cw.add_line(r'        "        }\n"')
        cw.add_line(r'        "    }\n"')
        cw.add_line(r'        "    \n"')
        cw.add_line(r'        "    // BatchNormalization + Step function\n"')
        cw.add_line(r'        "    for(int m = 0; m < M[0]; m++)\n"')
        cw.add_line(r'        "    {\n"')
        cw.add_line(r'        "        // BatchNormalization\n"')
        cw.add_line(r'        "        localOut[m] = localOut[m] * scale[m] + batchBias[m];\n"')
        cw.add_line(r'        "        // Step function\n"')
        cw.add_line(r'        "        if (localOut[m] > 0)\n"')
        cw.add_line(r'        "        {\n"')
        cw.add_line(r'        "            localOut[m] = 1.0;\n"')
        cw.add_line(r'        "        }\n"')
        cw.add_line(r'        "        else\n"')
        cw.add_line(r'        "        {\n"')
        cw.add_line(r'        "            localOut[m] = -1.0;\n"')
        cw.add_line(r'        "        }\n"')
        cw.add_line(r'        "    }\n"')
        cw.add_line(r'        "    \n"')
        cw.add_line(r'        "    // Write results into global output \n"')
        cw.add_line(r'        "    for(int m = 0; m < M[0]; m++)\n"')
        cw.add_line(r'        "    {\n"')
        cw.add_line(r'        "        output1[outNum + m] = (int)localOut[m];\n"')
        cw.add_line(r'        "    }\n"')
        cw.add_line(r'        "}";')
        cw.add_line(r'    size_t source_size1 = ' + str(1582 + len(str(channels))) + r';')
        cw.add_line(" ")

        cw.add_line(r'    char* source_str2 = (char*)malloc(' + str(1038 + len(str(channels))) + r' * sizeof(char));')
        cw.add_line(r'    source_str2 =')
        cw.add_line(r'        "__kernel void cnn_2(__global int* W0, __global int* W1, __global int* C,'
                    r' __global int* output0, __global int* output1)\n"')
        cw.add_line(r'        "{\n"')
        cw.add_line(r'        "    // Get the index of the current element\n"')
        cw.add_line(r'        "    int gid = get_global_id(0);\n"')
        cw.add_line(r'        "    \n"')
        cw.add_line(r'        "    // Local variables\n"')
        cw.add_line(r'        "    int localOut[' + str(channels) + r']; //Size of channels\n"')
        cw.add_line(r'        "    int tempOut;\n"')
        cw.add_line(r'        "    int outNum = C[0] * gid;\n"')
        cw.add_line(r'        "    int calcHelp = outNum;\n"')
        cw.add_line(r'        "    int calcH = outNum / (W1[0] * C[0]);\n"')
        cw.add_line(r'        "    calcHelp -= (calcH * W1[0] * C[0]);\n"')
        cw.add_line(r'        "    int calcW = calcHelp / C[0];\n"')
        cw.add_line(r'        "    \n"')
        cw.add_line(r'        "    // MaxPool\n"')
        cw.add_line(r'        "    for (int c = 0; c < C[0]; c++) {\n"')
        cw.add_line(r'        "        localOut[c] = -1;\n"')
        cw.add_line(r'        "    }\n"')
        cw.add_line(r'        "    for (int kH = 0; kH < 2; kH++) {\n"')
        cw.add_line(r'        "        for (int kW = 0; kW < 2; kW++) {\n"')
        cw.add_line(r'        "            for (int c = 0; c < C[0]; c++) {\n"')
        cw.add_line(r'        "                tempOut = output0[c + (calcW * 2 + kW) * (C[0]) +'
                    r' (calcH * 2 + kH) * (C[0] * W0[0])];\n"')
        cw.add_line(r'        "                if (tempOut > localOut[c]) {\n"')
        cw.add_line(r'        "                    localOut[c] = tempOut;\n"')
        cw.add_line(r'        "                }\n"')
        cw.add_line(r'        "            }\n"')
        cw.add_line(r'        "        }\n"')
        cw.add_line(r'        "    }\n"')
        cw.add_line(r'        "    \n"')
        cw.add_line(r'        "    // Write results into global output \n"')
        cw.add_line(r'        "    for(int c = 0; c < C[0]; c++)\n"')
        cw.add_line(r'        "    {\n"')
        cw.add_line(r'        "        output1[outNum + c] = localOut[c];\n"')
        cw.add_line(r'        "    }\n"')
        cw.add_line(r'        "}";')
        cw.add_line(r'    size_t source_size2 = ' + str(1038 + len(str(channels))) + r';')
        cw.add_line(" ")

        cw.add_line(r'    char* source_str3 = (char*)malloc(979 * sizeof(char));')
        cw.add_line("    source_str3 =")
        cw.add_line(
            r'        "__kernel void cnn_3(__global int* output0, __global int* OUTPUT0, __global int* output1, '
            r'__global char* batchNorm, __global double* bias, __global double* weight, __global double* batchBias, '
            r'__global double* scale)\n" '
        )
        cw.add_line(r'        "{\n" ')
        cw.add_line(r'        "    // Get the index of the current element\n" ')
        cw.add_line(r'        "    int gid = get_global_id(0);\n" ')
        cw.add_line(r'        "   \n" ')
        cw.add_line(r'        "   // Local variables\n" ')
        cw.add_line(r'        "   int weightNum = OUTPUT0[0] * gid;\n" ')
        cw.add_line(r'        "   double localOut;\n" ')
        cw.add_line(r'        "   \n" ')
        cw.add_line(r'        "   // Gemm\n" ')
        cw.add_line(r'        "   for (int i = 0; i < OUTPUT0[0]; i++)\n" ')
        cw.add_line(r'        "   {\n" ')
        cw.add_line(r'        "       if (i == 0)\n" ')
        cw.add_line(r'        "       {\n" ')
        cw.add_line(r'        "           localOut += bias[gid];\n" ')
        cw.add_line(r'        "       }\n" ')
        cw.add_line(r'        "       localOut += output0[i] * weight[weightNum + i];\n" ')
        cw.add_line(r'        "   }\n" ')
        cw.add_line(r'        "   \n" ')
        cw.add_line(r'        "   // BatchNormalization + Step function\n" ')
        cw.add_line(r'        "   if (batchNorm[0] == 1)\n" ')
        cw.add_line(r'        "   {\n" ')
        cw.add_line(r'        "       // BatchNormalization\n" ')
        cw.add_line(r'        "       localOut = localOut * scale[gid] + batchBias[gid];\n" ')
        cw.add_line(r'        "       // Step function\n" ')
        cw.add_line(r'        "       if (localOut > 0)\n" ')
        cw.add_line(r'        "       {\n" ')
        cw.add_line(r'        "           localOut = 1.0;\n" ')
        cw.add_line(r'        "       }\n" ')
        cw.add_line(r'        "       else\n" ')
        cw.add_line(r'        "       {\n" ')
        cw.add_line(r'        "           localOut = -1.0;\n" ')
        cw.add_line(r'        "       }\n" ')
        cw.add_line(r'        "   }\n" ')
        cw.add_line(r'        "   \n" ')
        cw.add_line(r'        "   // Write results into global output\n" ')
        cw.add_line(r'        "   output1[gid] = (int)localOut;\n" ')
        cw.add_line(r'        "   printf(\"\");" ')
        cw.add_line(r'        "}"; ')
        cw.add_line("    size_t source_size3 = 979;")
    else:
        cw.add_line("    // Load the kernel source code into the array source_str")
        cw.add_line(r'    char* source_str = (char*)malloc(979 * sizeof(char));')
        cw.add_line("    source_str =")
        cw.add_line(
            r'        "__kernel void bnn_1(__global int* output0, __global int* OUTPUT0, __global int* output1, '
            r'__global char* batchNorm, __global double* bias, __global double* weight, __global double* batchBias, '
            r'__global double* scale)\n" '
        )
        cw.add_line(r'        "{\n" ')
        cw.add_line(r'        "    // Get the index of the current element\n" ')
        cw.add_line(r'        "    int gid = get_global_id(0);\n" ')
        cw.add_line(r'        "   \n" ')
        cw.add_line(r'        "   // Local variables\n" ')
        cw.add_line(r'        "   int weightNum = OUTPUT0[0] * gid;\n" ')
        cw.add_line(r'        "   double localOut;\n" ')
        cw.add_line(r'        "   \n" ')
        cw.add_line(r'        "   // Gemm\n" ')
        cw.add_line(r'        "   for (int i = 0; i < OUTPUT0[0]; i++)\n" ')
        cw.add_line(r'        "   {\n" ')
        cw.add_line(r'        "       if (i == 0)\n" ')
        cw.add_line(r'        "       {\n" ')
        cw.add_line(r'        "           localOut += bias[gid];\n" ')
        cw.add_line(r'        "       }\n" ')
        cw.add_line(r'        "       localOut += output0[i] * weight[weightNum + i];\n" ')
        cw.add_line(r'        "   }\n" ')
        cw.add_line(r'        "   \n" ')
        cw.add_line(r'        "   // BatchNormalization + Step function\n" ')
        cw.add_line(r'        "   if (batchNorm[0] == 1)\n" ')
        cw.add_line(r'        "   {\n" ')
        cw.add_line(r'        "       // BatchNormalization\n" ')
        cw.add_line(r'        "       localOut = localOut * scale[gid] + batchBias[gid];\n" ')
        cw.add_line(r'        "       // Step function\n" ')
        cw.add_line(r'        "       if (localOut > 0)\n" ')
        cw.add_line(r'        "       {\n" ')
        cw.add_line(r'        "           localOut = 1.0;\n" ')
        cw.add_line(r'        "       }\n" ')
        cw.add_line(r'        "       else\n" ')
        cw.add_line(r'        "       {\n" ')
        cw.add_line(r'        "           localOut = -1.0;\n" ')
        cw.add_line(r'        "       }\n" ')
        cw.add_line(r'        "   }\n" ')
        cw.add_line(r'        "   \n" ')
        cw.add_line(r'        "   // Write results into global output\n" ')
        cw.add_line(r'        "   output1[gid] = (int)localOut;\n" ')
        cw.add_line(r'        "   printf(\"\");" ')
        cw.add_line(r'        "}"; ')
        cw.add_line("    size_t source_size = 979;")
    cw.add_line(" ")

    cw.add_line("    for (int i = 0; i < numInputs; i++)")
    cw.add_line("    {")
    if isCNN:
        cw.add_line("        args[i].source_str1 = source_str1;")
        cw.add_line("        args[i].source_str2 = source_str2;")
        cw.add_line("        args[i].source_str3 = source_str3;")
        cw.add_line("        args[i].source_size1 = source_size1;")
        cw.add_line("        args[i].source_size2 = source_size2;")
        cw.add_line("        args[i].source_size3 = source_size3;")
    else:
        cw.add_line("        args[i].source_str = source_str;")
        cw.add_line("        args[i].source_size = source_size;")
    cw.add_line("    }")
    cw.add_line(" ")

    cw.add_line("    // Malloc for biases, weights, batchBiases and scales for BNN")
    if isCNN:
        for i in range(int(numLayers)):
            if i != int(numLayers - 1):
                cw.add_line(
                    "    double* bias" + str(i + 1) + " = (double*)malloc(sizeof(double) * CHANNELS);")
            else:
                cw.add_line(
                    "    double* bias" + str(i + 1) + " = (double*)malloc(sizeof(double) * OUTPUT" + str(i + 3) + ");")
        for i in range(int(numLayers)):
            cw.add_line(
                "    double* weight" + str(i + 1) + " = (double*)malloc(sizeof(double) * WEIGHT" + str(i + 1) + ");")
        for i in range(int(numLayers - 1)):
            cw.add_line(
                "    double* batchBias" + str(i + 1) + " = (double*)malloc(sizeof(double) * CHANNELS);")
        for i in range(int(numLayers - 1)):
            cw.add_line(
                "    double* scale" + str(i + 1) + " = (double*)malloc(sizeof(double) * CHANNELS);")
    else:
        bnnMalloc = [" " for i in range(int(dataKeys))]
        for i in range(int(numLayers)):
            bnnMalloc[i] = "".join(
                ("    double* bias", str(i + 1), " = (double*)malloc(sizeof(double) * OUTPUT", str(i + 1), ");"))
            bnnMalloc[int(i + numLayers)] = "".join(
                ("    double* weight", str(i + 1), " = (double*)malloc(sizeof(double) * OUTPUT", str(i + 1),
                 " * OUTPUT", str(i), ");"))
            if i != int(numLayers - 1):
                bnnMalloc[int(i + 2 * numLayers)] = "".join(
                    ("    double* batchBias", str(i + 1), " = (double*)malloc(sizeof(double) * OUTPUT", str(i + 1),
                     ");"))
                bnnMalloc[int(i + 3 * numLayers - 1)] = "".join(
                    ("    double* scale", str(i + 1), " = (double*)malloc(sizeof(double) * OUTPUT", str(i + 1), ");"))
        cw.add_lines(bnnMalloc)
    cw.add_line(" ")

    # Load BNN data

    cw.add_line("    // Json object for the data")
    cw.add_line("    json_object* json_biases;")
    cw.add_line("    json_object* json_weights;")
    cw.add_line("    json_object* json_batchBiases;")
    cw.add_line("    json_object* json_scales;")
    cw.add_line(" ")

    cw.add_line("    json_object* parsed_data;")
    cw.add_line("    char* dataBuffer = (char*)malloc(JSON_SIZE * sizeof(char));")
    cw.add_line(" ")

    cw.add_line("    // Read the data for the BNN")
    cw.add_line("    FILE* dataBNN;")
    cw.add_line(r'    dataBNN = fopen("' + nameJson + r'", "r");')
    cw.add_line("    if (!dataBNN)")
    cw.add_line("    {")
    cw.add_line(r'        fprintf(stderr, "Failed to load data for the BNN.\n");')
    cw.add_line("        exit(1);")
    cw.add_line("    }")
    cw.add_line("    fread(dataBuffer, JSON_SIZE, 1, dataBNN);")
    cw.add_line("    fclose(dataBNN);")
    cw.add_line("    parsed_data = json_tokener_parse(dataBuffer);")
    cw.add_line(" ")
    cw.add_line(r'    json_object_object_get_ex(parsed_data, "biases", &json_biases);')
    cw.add_line(r'    json_object_object_get_ex(parsed_data, "weights", &json_weights);')
    cw.add_line(r'    json_object_object_get_ex(parsed_data, "batchBiases", &json_batchBiases);')
    cw.add_line(r'    json_object_object_get_ex(parsed_data, "scales", &json_scales);')
    cw.add_line(" ")

    cw.add_line("    // Write the data from json into malloced memory")
    if isCNN:
        for i in range(int(numLayers)):
            cw.add_line("    for (int i = 0; i < WEIGHT" + str(i + 1) + "; i++)")
            cw.add_line("    {")
            if i != int(numLayers - 1):
                cw.add_line("        if (i < CHANNELS)")
            else:
                cw.add_line("        if (i < OUTPUT" + str(i + 3) + ")")
            cw.add_line("        {")
            cw.add_line(
                "            bias" + str(i + 1) + "[i] = json_object_get_double(json_object_array_get_idx("
                "json_object_array_get_idx(json_biases, " + str(i) + "), i));"
            )
            if i != int(numLayers - 1):
                cw.add_line(
                    "            batchBias" + str(i + 1) + "[i] = json_object_get_double(json_object_array_get_idx("
                    "json_object_array_get_idx(json_batchBiases, " + str(i) + "), i));"
                )
                cw.add_line(
                    "            scale" + str(i + 1) + "[i] = json_object_get_double(json_object_array_get_idx("
                    "json_object_array_get_idx(json_scales, " + str(i) + "), i));"
                )
            cw.add_line("        }")
            cw.add_line(
                "        weight" + str(i + 1) + "[i] = json_object_get_double(json_object_array_get_idx("
                "json_object_array_get_idx(json_weights, " + str(i) + "), i));"
            )
            cw.add_line("    }")
    else:
        for i in range(int(numLayers)):
            cw.add_line("    for (int i = 0; i < OUTPUT" + str(i + 1) + " * OUTPUT" + str(i) + "; i++)")
            cw.add_line("    {")
            cw.add_line("        if (i < OUTPUT" + str(i + 1) + ")")
            cw.add_line("        {")
            cw.add_line(
                "            bias" + str(i + 1) + "[i] = json_object_get_double(json_object_array_get_idx("
                "json_object_array_get_idx(json_biases, " + str(i) + "), i));"
            )
            if i != int(numLayers - 1):
                cw.add_line(
                    "            batchBias" + str(i + 1) + "[i] = json_object_get_double(json_object_array_get_idx("
                    "json_object_array_get_idx(json_batchBiases, " + str(i) + "), i));"
                )
                cw.add_line(
                    "            scale" + str(i + 1) + "[i] = json_object_get_double(json_object_array_get_idx("
                    "json_object_array_get_idx(json_scales, " + str(i) + "), i));"
                )
            cw.add_line("        }")
            cw.add_line(
                "        weight" + str(i + 1) + "[i] = json_object_get_double(json_object_array_get_idx("
                "json_object_array_get_idx(json_weights, " + str(i) + "), i));"
            )
            cw.add_line("    }")
    cw.add_line(" ")

    cw.add_line("    for (int i = 0; i < numInputs; i++)")
    cw.add_line("    {")
    for i in range(int(numLayers)):
        cw.add_line("        args[i].bias" + str(i + 1) + " = bias" + str(i + 1) + ";")
        cw.add_line("        args[i].weight" + str(i + 1) + " = weight" + str(i + 1) + ";")
        if i != int(numLayers - 1):
            cw.add_line("        args[i].batchBias" + str(i + 1) + " = batchBias" + str(i + 1) + ";")
            cw.add_line("        args[i].scale" + str(i + 1) + " = scale" + str(i + 1) + ";")
    cw.add_line("    }")
    cw.add_line(" ")

    cw.add_line(r'    // Create threads')
    cw.add_line(r'    pthread_t* threads = NULL;')
    cw.add_line(r'    if (parallel == 0)')
    cw.add_line(r'    {')
    cw.add_line(r'        threads = (pthread_t*)malloc(sizeof(pthread_t) * numInputs);')
    cw.add_line(r'    }')
    cw.add_line(r' ')
    cw.add_line(r'    // Calculate the output for every input')
    cw.add_line(r'    for (int i = 0; i < numInputs; i++)')
    cw.add_line(r'    {')
    cw.add_line(r'        args[i].output0 = (int*)malloc(sizeof(int) * OUTPUT0);')
    cw.add_line(r'        for (int j = 0; j < OUTPUT0; j++)')
    cw.add_line(r'        {  ')
    cw.add_line(r'            args[i].output0[j] = json_object_get_int(json_object_array_get_idx('
                r'json_object_array_get_idx(json_inputs, i), j));')
    cw.add_line(r'        }')
    cw.add_line(r'        if (parallel == 0)')
    cw.add_line(r'        {')
    cw.add_line(r'            pthread_create(&threads[i], NULL, &calcBNN, (void*)&args[i]);')
    cw.add_line(r'        }')
    cw.add_line(r'        else')
    cw.add_line(r'        {')
    cw.add_line(r'            calcBNN((void*)&args[i]);')
    cw.add_line(r'        }')
    cw.add_line(r'    }')
    cw.add_line(r' ')
    cw.add_line(r'    // Clean up')
    cw.add_line(r'    if (parallel == 0)')
    cw.add_line(r'    {')
    cw.add_line(r'        for (int i = 0; i < numInputs; i++)')
    cw.add_line(r'        {')
    cw.add_line(r'            pthread_join(threads[i], NULL);')
    cw.add_line(r'        }')
    cw.add_line(r'        free(threads);')
    cw.add_line(r'    }')
    cw.add_line(r' ')

    cw.add_line(r'    free(inputsBuffer);')
    cw.add_line(r'    free(dataBuffer);')
    for i in range(int(numLayers)):
        cw.add_line("    free(bias" + str(i + 1) + ");")
        cw.add_line("    free(weight" + str(i + 1) + ");")
        if i != int(numLayers - 1):
            cw.add_line("    free(batchBias" + str(i + 1) + ");")
            cw.add_line("    free(scale" + str(i + 1) + ");")
    cw.add_line(r'    for (int i = 0; i < numInputs; i++)')
    cw.add_line(r'    {')
    cw.add_line(r'        free(args[i].output0);')
    cw.add_line(r'        free(args[i].nameOutput);')
    cw.add_line(r'        free(args[i].jsonOutput);')
    cw.add_line(r'    }')
    cw.add_line(r'    free(args);')
    cw.add_line(r' ')

    cw.add_line(r'    printf("Created output data.\n");')
    cw.add_line(r' ')

    cw.add_line("    return 0;")
    cw.add_line("}")

    f = open(nameC, "w")
    f.write(str(cw))
    f.close()

    print("Wrote C-code into " + nameC)

    # print(cw)

except ValueError as e:
    print("JSON-file couldn't be opened.")
    print("Error: " + str(e))
except FileNotFoundError as e:
    print("JSON-file couldn't be found.")
    print("Error: " + str(e))
except KeyError as e:
    print(r'Keys are missing in JSON-file. Needs Keys: "biases", "weights", "batchBiases" and "scales"')
    print("Error: " + str(e))
