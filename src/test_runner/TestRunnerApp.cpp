#include <string>
#include <cassert>
#include <stdexcept>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <gl/glew.h>
#include <windowsx.h>
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>

#include <math/matrix.h>
#include <pga/core/GlobalVariables.cuh>
#include <pga/core/CUDAException.h>
#include <pga/rendering/RenderingGlobalVariables.cuh>
#include <pga/rendering/GLException.h>
#include <pga/rendering/Texture.h>
#include <pga/rendering/PNG.h>
#include <pga/rendering/InstancedTriangleMeshSource.h>
#include <pga/rendering/ShapeMesh.h>
#include <pga/rendering/OBJMesh.h>
#include <pga/rendering/Shader.h>
#include <pga/rendering/ColorShader.h>
#include <pga/rendering/TexturedShader.h>

#include "PGAFacade.h"
#include "TestRunnerApp.h"

//////////////////////////////////////////////////////////////////////////
enum ReturnValue
{
	RV_SUCCESS = 0,
	RV_INVALID_PARAMETERS,
	RV_GLEW_INITIALIZATION_ERROR,
	RV_CREATE_DIRECTORY_ERROR,
	RV_WINDOWS_API_ERROR,
	RV_CUDA_EXCEPTION,
	RV_GL_EXCEPTION,
	RV_GEOMETRY_BUFFER_OVERFLOW_EXCEPTION,
	RV_RUNTIME_ERROR,
	RV_UNEXPECTED_ERROR

};

//////////////////////////////////////////////////////////////////////////
#define win32Assert(resultHandle, errorMessage) \
	if (resultHandle == 0) \
		{ \
			std::cerr << ##errorMessage << std::endl; \
			dispose(); \
			exit(RV_WINDOWS_API_ERROR); \
		} \

//////////////////////////////////////////////////////////////////////////
class GeneratorBufferOverflow : public std::exception {};

//////////////////////////////////////////////////////////////////////////
std::unique_ptr<PGA::Rendering::Texture2D> loadTexture2D(std::string& fileName)
{
	std::string filenameExtension = fileName.substr(fileName.rfind('.') + 1, fileName.size() - 1);
	std::transform(filenameExtension.begin(), filenameExtension.end(), filenameExtension.begin(), ::tolower);
	if (filenameExtension == "png")
		return std::unique_ptr<PGA::Rendering::Texture2D>(new PGA::Rendering::Texture2D(PGA::Rendering::PNG::loadTexture2D(fileName, true), true));
	else
		return nullptr;
}

//////////////////////////////////////////////////////////////////////////
TestRunnerApp* TestRunnerApp::s_instance = 0;
const char* TestRunnerApp::WINDOW_TITLE = "test_runner";
const char* TestRunnerApp::WINDOW_CLASS_NAME = "test_runner_window";
const unsigned int TestRunnerApp::SCREEN_WIDTH = 1024;
const unsigned int TestRunnerApp::SCREEN_HEIGHT = 768;
const unsigned int TestRunnerApp::BYTES_PER_PIXEL = 4;
const unsigned int TestRunnerApp::COLOR_BUFFER_BITS = 32;
const unsigned int TestRunnerApp::DEPTH_BUFFER_BITS = 32;
const unsigned int TestRunnerApp::HAS_ALPHA = 0;
const PIXELFORMATDESCRIPTOR TestRunnerApp::PIXEL_FORMAT_DESCRIPTOR = { sizeof(PIXELFORMATDESCRIPTOR), 1, PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER, PFD_TYPE_RGBA, COLOR_BUFFER_BITS, 0, 0, 0, 0, 0, 0, HAS_ALPHA, 0, 0, 0, 0, 0, 0, DEPTH_BUFFER_BITS, 0, 0, PFD_MAIN_PLANE, 0, 0, 0, 0 };
const float TestRunnerApp::ANGLE_INCREMENT = 0.05f;
const float TestRunnerApp::CAMERA_PITCH_LIMIT = 1.0472f; // 60 deg.
const float TestRunnerApp::CAMERA_MOVE_SPEED = 100.0f;

//////////////////////////////////////////////////////////////////////////
TestRunnerApp::TestRunnerApp() :
	applicationHandle(0),
	windowHandle(0),
	deviceContextHandle(0),
	pixelFormat(0),
	openGLRenderingContextHandle(0),
	camera(SCREEN_WIDTH / (float)SCREEN_HEIGHT, 60, 0.1f, 1000.0f, math::float3(0, 0, -10.0f)),
	mouseButtonPressed(false),
	lastMousePosition(-1, -1),
	cameraUniformBuffer(0),
	cameraPhi(-math::constants<float>::pi() * 0.5f),
	cameraTheta(0),
	generator(nullptr),
	generatePGASeedsEveryFrame(false),
	generateGeometryEveryFrame(false),
	silent(false)
{
	memset(keys, 0, sizeof(bool) * 0xff);
	memset(pressedKeys, 0, sizeof(bool) * 0xff);
	s_instance = this;
}

//////////////////////////////////////////////////////////////////////////
TestRunnerApp::~TestRunnerApp()
{
	s_instance = nullptr;
}

//////////////////////////////////////////////////////////////////////////
double TestRunnerApp::generateGeometry()
{
	if (generator == nullptr)
		createGenerator();
	generator->bind();
	auto generationTime = executePGA();
	if (hasBufferOverflow())
		throw GeneratorBufferOverflow();
	generator->unbind();
	return generationTime;
}

//////////////////////////////////////////////////////////////////////////
void TestRunnerApp::createMaterial(int materialRef)
{
	auto it = configuration.materials.find(materialRef);
	if (it == configuration.materials.end())
		return;
	auto& material = it->second;
	std::unique_ptr<PGA::Rendering::Material> newMaterial(new PGA::Rendering::Material());
	switch (material.type)
	{
	case 0: // Color
		newMaterial->setShader(
			std::unique_ptr<PGA::Rendering::ColorShader>(new PGA::Rendering::ColorShader())
			);
		if (material.hasAttribute("color0"))
			newMaterial->setParameter("color0", material.getFloat4Attribute("color0"));
		else
			std::cout << "[WARNING] material parameter color0 not set [materialRef=" << std::to_string(materialRef) << "]" << std::endl;
		break;
	case 1: // Textured
		newMaterial->setShader(
			std::unique_ptr<PGA::Rendering::TexturedShader>(new PGA::Rendering::TexturedShader())
			);
		if (material.hasAttribute("tex0"))
		{
			auto textureFilename = configuration.textureRootPath + material.getAttribute("tex0");
			auto texture = loadTexture2D(textureFilename);
			if (texture != nullptr)
				newMaterial->setParameter("tex0", std::move(texture));
			else
				std::cout << "[WARNING] could not load texture [materialRef=" << std::to_string(materialRef) << ", textureFilename=\"" << textureFilename << "\"]" << std::endl;
		}
		else
			std::cout << "[WARNING] material parameter tex0 not set [materialRef=" << std::to_string(materialRef) << "]" << std::endl;
		if (material.hasAttribute("useUvX0"))
			newMaterial->setParameter("useUvX0", material.getFloatAttribute("useUvX0"));
		else
			newMaterial->setParameter("useUvX0", 1.0f);
		if (material.hasAttribute("useUvY0"))
			newMaterial->setParameter("useUvY0", material.getFloatAttribute("useUvY0"));
		else
			newMaterial->setParameter("useUvY0", 1.0f);
		if (material.hasAttribute("uvScale0"))
			newMaterial->setParameter("uvScale0", material.getFloat2Attribute("uvScale0"));
		else
			newMaterial->setParameter("uvScale0", math::float2(1.0f, 1.0f));
		break;
	default:
		throw std::runtime_error("unknown material type [materialRef=" + std::to_string(materialRef) + ", configMaterial.type=" + std::to_string(material.type) + "]");
	}
	newMaterial->setBackFaceCulling(material.backFaceCulling);
	materials.emplace(materialRef, std::move(newMaterial));
}

//////////////////////////////////////////////////////////////////////////
std::vector<std::unique_ptr<PGA::Rendering::TriangleMesh>> TestRunnerApp::createTriangleMeshes()
{
	std::vector<std::unique_ptr<PGA::Rendering::TriangleMesh>> triangleMeshes;
	std::string textureFilename;
	std::unique_ptr<PGA::Rendering::Texture> texture;
	for (auto& entry : configuration.triangleMeshes)
	{
		auto& triangleMesh = entry.second;
		createMaterial(triangleMesh.materialRef);
		triangleMeshes.emplace_back(
			new PGA::Rendering::TriangleMesh(
				triangleMesh.maxNumVertices,
				triangleMesh.maxNumIndices
			)
		);
	}
	return triangleMeshes;
}

//////////////////////////////////////////////////////////////////////////
std::vector<std::unique_ptr<PGA::Rendering::InstancedTriangleMesh>> TestRunnerApp::createInstancedTriangleMeshes()
{
	std::vector<std::unique_ptr<PGA::Rendering::InstancedTriangleMesh>> instancedTriangleMeshes;
	for (auto& entry : configuration.instancedTriangleMeshes)
	{
		auto& instancedTriangleMesh = entry.second;
		createMaterial(instancedTriangleMesh.materialRef);
		std::unique_ptr<PGA::Rendering::InstancedTriangleMeshSource> source;
		switch (instancedTriangleMesh.type)
		{
		case PGA::Rendering::Configuration::InstancedTriangleMesh::SHAPE:
			if (instancedTriangleMesh.shape == "Quad")
			{
				source = std::unique_ptr<PGA::Rendering::ShapeMesh<PGA::Shapes::Quad>>(
					new PGA::Rendering::ShapeMesh<PGA::Shapes::Quad>()
					);
			}
			else if (instancedTriangleMesh.shape == "DynamicConvexPolygon")
			{
				source = std::unique_ptr<PGA::Rendering::ShapeMesh<DCPoly>>(
					new PGA::Rendering::ShapeMesh<DCPoly>(DCPoly(&instancedTriangleMesh.vertices[0], static_cast<unsigned int>(instancedTriangleMesh.vertices.size())))
					);
			}
			else if (instancedTriangleMesh.shape == "DynamicPolygon")
			{
				source = std::unique_ptr<PGA::Rendering::ShapeMesh<DPoly>>(
					new PGA::Rendering::ShapeMesh<DPoly>(DPoly(&instancedTriangleMesh.vertices[0], static_cast<unsigned int>(instancedTriangleMesh.vertices.size())))
					);
			}
			else if (instancedTriangleMesh.shape == "Box")
			{
				source = std::unique_ptr<PGA::Rendering::ShapeMesh<PGA::Shapes::Box>>(
					new PGA::Rendering::ShapeMesh<PGA::Shapes::Box>()
					);
			}
			else if (instancedTriangleMesh.shape == "DynamicConvexRightPrism")
			{
				source = std::unique_ptr<PGA::Rendering::ShapeMesh<DCRPrism>>(
					new PGA::Rendering::ShapeMesh<DCRPrism>(DCRPrism(&instancedTriangleMesh.vertices[0], static_cast<unsigned int>(instancedTriangleMesh.vertices.size())))
					);
			}
			else if (instancedTriangleMesh.shape == "DynamicRightPrism")
			{
				source = std::unique_ptr<PGA::Rendering::ShapeMesh<DRPrism>>(
					new PGA::Rendering::ShapeMesh<DRPrism>(DRPrism(&instancedTriangleMesh.vertices[0], static_cast<unsigned int>(instancedTriangleMesh.vertices.size())))
					);
			}
			else if (instancedTriangleMesh.shape == "Sphere")
			{
				source = std::unique_ptr<PGA::Rendering::ShapeMesh<PGA::Shapes::Sphere>>(
					new PGA::Rendering::ShapeMesh<PGA::Shapes::Sphere>()
					);
			}
			else
				std::cout << "[WARNING] unknown shape instanced mesh shape type [instancedTriangleMesh.shape=\"" << instancedTriangleMesh.shape << "\"]" << std::endl;
			break;
		case PGA::Rendering::Configuration::InstancedTriangleMesh::OBJ:
			source = std::unique_ptr<PGA::Rendering::OBJMesh>(
				new PGA::Rendering::OBJMesh(configuration.modelRootPath + instancedTriangleMesh.modelPath)
				);
			break;
		default:
			throw std::runtime_error("unknown instanced triangle mesh type");
		}
		instancedTriangleMeshes.emplace_back(new PGA::Rendering::InstancedTriangleMesh(
			instancedTriangleMesh.maxNumElements,
			std::move(source)
			)
			);
	}
	return instancedTriangleMeshes;
}

//////////////////////////////////////////////////////////////////////////
bool TestRunnerApp::hasBufferOverflow()
{
	bool overflow = false;
	for (auto i = 0; i < generator->getNumTriangleMeshes(); i++)
	{
		const auto& triangleMesh = generator->getTriangleMesh(i);
		if (triangleMesh->hasOverflown())
		{
			std::cout << "[WARNING] triangle mesh " << i <<
				" no. vertices and/or no. indices have overflown " <<
				"(max. vert./curr. vert):  " << triangleMesh->getMaxNumVertices() << " / " << triangleMesh->getNumVertices() <<
				", " <<
				"(max. ind./curr. ind): " << triangleMesh->getMaxNumIndices() << " / " << triangleMesh->getNumIndices() <<
				std::endl;
			overflow = true;
		}
	}
	for (auto i = 0; i < generator->getNumInstancedTriangleMeshes(); i++)
	{
		const auto& instancedTriangleMesh = generator->getInstancedTriangleMesh(i);
		if (instancedTriangleMesh->hasOverflow())
		{
			std::cout << "[WARNING] instanced triangle mesh " << i << " no. elements has overflown (max/current): " <<
				instancedTriangleMesh->getMaxNumInstances() << " / " <<
				instancedTriangleMesh->getNumInstances() << std::endl;
			overflow = true;
		}
	}
	return overflow;
}

//////////////////////////////////////////////////////////////////////////
// This method exists because we cannot guarantee that resources 
// from neither GL or CUDA can be deallocated, so we simply release them
void TestRunnerApp::finalizePGAWithException()
{
	for (auto& material : materials)
		material.second.release();
	generator.release();
	releasePGA();
}

//////////////////////////////////////////////////////////////////////////
int TestRunnerApp::run(unsigned int argc, const char** argv)
{
	//////////////////////////////////////////////////////////////////////////
	// Read command line arguments

	int deviceIndex = argc > 1 ? atoi(argv[1]) : 0;
	long globalSeed = argc > 2 ? atol(argv[2]) : -1;
	bool autoMode = argc > 3 ? (std::string(argv[3]) == "auto") : false;
	silent = argc > 4 ? (std::string(argv[4]) == "true") : false;

	int numRuns = 0;
	int minNumElements = 0;
	int maxNumElements = 0;
	int elementsStep = 0;
    int counter;
    int currElems;
	if (autoMode)
	{
		if (argc < 9)
		{
			std::cerr << "automatic mode requires 4 extra parameters: numRuns minElements maxElements elementsStep" << std::endl;
			exit(RV_INVALID_PARAMETERS);
		}

		numRuns = atoi(argv[5]);
		minNumElements = atoi(argv[6]);
		maxNumElements = atoi(argv[7]);
		elementsStep = atoi(argv[8]);

		if (numRuns < 1)
		{
			std::cerr << "numRums must be > 1" << std::endl;
			exit(RV_INVALID_PARAMETERS);
		}

		if (minNumElements < 0)
		{
			std::cerr << "minElements must be >= 0" << std::endl;
			exit(RV_INVALID_PARAMETERS);
		}

		if (maxNumElements < minNumElements)
		{
			std::cerr << "maxElements >= minElements" << std::endl;
			exit(RV_INVALID_PARAMETERS);
		}

		if (elementsStep < 1)
		{
			std::cerr << "elementsStep must be > 1" << std::endl;
			exit(RV_INVALID_PARAMETERS);
		}
	}

	//////////////////////////////////////////////////////////////////////////
	// Setting up pseudo-random number generator

	if (globalSeed >= 0)
		prng.seed((unsigned long)globalSeed);
	else
		prng.seed((unsigned long)std::chrono::system_clock::now().time_since_epoch().count());

	//////////////////////////////////////////////////////////////////////////
	// Create window

	applicationHandle = GetModuleHandle(0);

	WNDCLASSEX windowClass;
	windowClass.cbSize = sizeof(WNDCLASSEX);
	windowClass.style = CS_HREDRAW | CS_VREDRAW | CS_OWNDC;
	windowClass.lpfnWndProc = WinProc;
	windowClass.cbClsExtra = 0;
	windowClass.cbWndExtra = 0;
	windowClass.hInstance = applicationHandle;
	windowClass.hIcon = LoadIcon(applicationHandle, MAKEINTRESOURCE(IDI_APPLICATION));
	windowClass.hCursor = LoadCursor(NULL, IDC_ARROW);
	windowClass.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
	windowClass.lpszMenuName = NULL;
	windowClass.lpszClassName = WINDOW_CLASS_NAME;
	windowClass.hIconSm = LoadIcon(windowClass.hInstance, MAKEINTRESOURCE(IDI_APPLICATION));

	win32Assert(RegisterClassEx(&windowClass), "RegisterClassEx failed");
	win32Assert((windowHandle = CreateWindow(WINDOW_CLASS_NAME, WINDOW_TITLE, (WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_CLIPSIBLINGS | WS_CLIPCHILDREN), CW_USEDEFAULT, CW_USEDEFAULT, SCREEN_WIDTH, SCREEN_HEIGHT, NULL, NULL, applicationHandle, NULL)), "CreateWindow failed");
	win32Assert((deviceContextHandle = GetDC(windowHandle)), "GetDC() failed");
	win32Assert((pixelFormat = ChoosePixelFormat(deviceContextHandle, &PIXEL_FORMAT_DESCRIPTOR)), "ChoosePixelFormat() failed");
	win32Assert(SetPixelFormat(deviceContextHandle, pixelFormat, &PIXEL_FORMAT_DESCRIPTOR), "SetPixelFormat() failed");
	win32Assert((openGLRenderingContextHandle = wglCreateContext(deviceContextHandle)), "wglCreateContext() failed");
	win32Assert(wglMakeCurrent(deviceContextHandle, openGLRenderingContextHandle), "wglMakeCurrent() failed");
	ShowWindow(windowHandle, SW_SHOW);
	SetForegroundWindow(windowHandle);
	SetFocus(windowHandle);

	//////////////////////////////////////////////////////////////////////////
	// Initialize GLEW

	if (glewInit())
	{
		std::cerr << "error initializing GLEW" << std::endl;
		DestroyWindow(windowHandle);
		dispose();
		exit(RV_GLEW_INITIALIZATION_ERROR);
	}

	int returnValue = RV_SUCCESS;
	try
	{
		//////////////////////////////////////////////////////////////////////////
		// One-time initialization

		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, deviceIndex);
		cudaGLSetGLDevice(deviceIndex);

		std::cout << "device name: " << deviceProp.name << std::endl;
		std::cout << "scene name: " << getSceneName() << std::endl << std::endl;

		PGA::Rendering::Shader::setIncludePath({ "shaders/" });

		initializePGA();
		generatePGASeeds();

		if (autoMode)
		{
			counter = 0;
			currElems = -1;

			std::string dataDir("data/");
			if (!CreateDirectory(dataDir.c_str(), 0) && GetLastError() != ERROR_ALREADY_EXISTS)
			{
				std::cerr << "error creating directory: " << dataDir << std::endl;
				exit(RV_CREATE_DIRECTORY_ERROR);
			}
			cudaDeviceProp deviceProps;
			cudaGetDeviceProperties(&deviceProps, deviceIndex);
			dataDir += std::string(deviceProps.name) + '/';
			if (!CreateDirectory(dataDir.c_str(), 0) && GetLastError() != ERROR_ALREADY_EXISTS)
			{
				std::cerr << "error creating directory: " << dataDir << std::endl;
				exit(RV_CREATE_DIRECTORY_ERROR);
			}
			dataDir += getSceneName() + '/';
			if (!CreateDirectory(dataDir.c_str(), 0) && GetLastError() != ERROR_ALREADY_EXISTS)
			{
				std::cerr << "error creating directory: " << dataDir << std::endl;
				exit(RV_CREATE_DIRECTORY_ERROR);
			}
			logManager.setBaseDir(dataDir);
		}
		else
			generateGeometry();

		glEnable(GL_DEPTH_TEST);
		glViewport(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT);

		glGenBuffers(1, &cameraUniformBuffer);

		//////////////////////////////////////////////////////////////////////////
		// Main application loop

		MSG message;
		while (true)
		{
			if (PeekMessage(&message, NULL, 0, 0, PM_REMOVE))
			{
				if (message.message == WM_QUIT)
				{
					break;
				}
				else
				{
					TranslateMessage(&message);
					DispatchMessage(&message);
				}
			}
			else
			{
				std::chrono::system_clock::duration start, end;
				if (generatePGASeedsEveryFrame)
					generatePGASeeds();
				if (autoMode)
				{
					auto elems = minNumElements + (elementsStep * ((int)counter++ / numRuns));
					if (elems > maxNumElements)
						break;
					if (elems != currElems)
					{
						setNumElements(elems);
						if (!silent)
						{
							std::string testName = getTestName();
							logManager.addLogger("auto", testName + "-auto.txt");
							logManager.addLogger("generation", testName + "-generation.txt");
							if (isInstrumented())
							{
								logManager.initializeForEdgeInstrumentation("edges", testName + "-edges.csv");
								logManager.initializeForCuptiInstrumentation("cupti", testName + "-cupti.txt");
								logManager.initializeForSubgraphInstrumentation("subgraphs", testName + "-subgraphs.csv");
							}
							logManager.write("auto", numRuns);
							logManager.write("auto", minNumElements);
							logManager.write("auto", maxNumElements);
							logManager.write("auto", elementsStep);
						}
						currElems = elems;
					}
					auto generationTime = generateGeometry();
					if (!silent)
						logManager.write("generation", generationTime);
				}
				else
				{
					start = std::chrono::system_clock::now().time_since_epoch();
					if (generateGeometryEveryFrame)
						generateGeometry();
					drawScene();
					end = std::chrono::system_clock::now().time_since_epoch();
				}
				float deltaTime = (std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000000.0f);
				if (autoMode && !silent)
					logManager.write("rendering", deltaTime);
				processKeys(deltaTime);
				memset(pressedKeys, 0, sizeof(bool) * 0xFF);
				std::stringstream stream;
				stream << std::fixed << std::setprecision(5) << WINDOW_TITLE << " @ fps: " << (1 / deltaTime);
				SetWindowText(windowHandle, stream.str().c_str());
			}
		}

		glDeleteBuffers(1, &cameraUniformBuffer);

		//////////////////////////////////////////////////////////////////////////
		// Finalize PGA
		materials.clear();
		generator = 0;
		destroyPGA();
	}
	catch (PGA::CUDA::Exception& e)
	{
		std::cerr << e.what() << std::endl;
		DestroyWindow(windowHandle);
		returnValue = RV_CUDA_EXCEPTION;
		finalizePGAWithException();
	}
	catch (PGA::Rendering::GL::Exception& e)
	{
		std::cerr << e.what() << std::endl;
		DestroyWindow(windowHandle);
		returnValue = RV_GL_EXCEPTION;
		finalizePGAWithException();
	}
	catch (GeneratorBufferOverflow&)
	{
		std::cerr << "generator buffer overflow" << std::endl;
		DestroyWindow(windowHandle);
		returnValue = RV_GEOMETRY_BUFFER_OVERFLOW_EXCEPTION;
		finalizePGAWithException();
	}
	catch (std::runtime_error& e)
	{
		std::cerr << e.what() << std::endl;
		DestroyWindow(windowHandle);
		returnValue = RV_RUNTIME_ERROR;
		finalizePGAWithException();
	}
	catch (...)
	{
		std::cerr << "unknown error" << std::endl;
		DestroyWindow(windowHandle);
		returnValue = RV_UNEXPECTED_ERROR;
		finalizePGAWithException();
	}

	if (silent)
	{
		std::ofstream file("run.log");
		file << getNumElements() << std::endl;
		file.close();
	}

	if (autoMode && isInstrumented())
		logManager.finalizeForInstrumentation();

	dispose();

	DestroyWindow(windowHandle);
	win32Assert(UnregisterClass(WINDOW_CLASS_NAME, applicationHandle), "UnregisterClass() failed");

	deviceContextHandle = 0;
	windowHandle = 0;
	applicationHandle = 0;

	return returnValue;
}

//////////////////////////////////////////////////////////////////////////
void TestRunnerApp::dispose()
{
	if (openGLRenderingContextHandle)
	{
		win32Assert(wglMakeCurrent(0, 0), "wglMakeCurrent() failed");
		win32Assert(wglDeleteContext(openGLRenderingContextHandle), "wglDeleteContext() failed");
		openGLRenderingContextHandle = 0;
	}
}

//////////////////////////////////////////////////////////////////////////
void TestRunnerApp::moveCameraLeft(float deltaTime)
{
	camera.position() -= camera.u() * deltaTime * CAMERA_MOVE_SPEED;
}

//////////////////////////////////////////////////////////////////////////
void TestRunnerApp::moveCameraRight(float deltaTime)
{
	camera.position() += camera.u() * deltaTime * CAMERA_MOVE_SPEED;
}

//////////////////////////////////////////////////////////////////////////
void TestRunnerApp::moveCameraForward(float deltaTime)
{
	camera.position() += camera.w() * deltaTime * CAMERA_MOVE_SPEED;
}

//////////////////////////////////////////////////////////////////////////
void TestRunnerApp::moveCameraBackward(float deltaTime)
{
	camera.position() -= camera.w() * deltaTime * CAMERA_MOVE_SPEED;
}

//////////////////////////////////////////////////////////////////////////
void TestRunnerApp::moveCameraUp(float deltaTime)
{
	camera.position() += camera.v() * deltaTime * CAMERA_MOVE_SPEED;
}

//////////////////////////////////////////////////////////////////////////
void TestRunnerApp::moveCameraDown(float deltaTime)
{
	camera.position() -= camera.v() * deltaTime * CAMERA_MOVE_SPEED;
}

//////////////////////////////////////////////////////////////////////////
void TestRunnerApp::drawScene()
{
	Camera::UniformBuffer cameraBuffer;
	camera.writeToBuffer(&cameraBuffer);
	glBindBuffer(GL_UNIFORM_BUFFER, cameraUniformBuffer);
	glBufferData(GL_UNIFORM_BUFFER, 0, 0, GL_STATIC_DRAW);
	glBufferData(GL_UNIFORM_BUFFER, sizeof(Camera::UniformBuffer), &cameraBuffer, GL_STATIC_DRAW);
	glBindBuffer(GL_UNIFORM_BUFFER, 0);
	glBindBufferRange(GL_UNIFORM_BUFFER, 0, cameraUniformBuffer, 0, sizeof(Camera::UniformBuffer));

	glClearColor(1, 1, 1, 1);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	if (generator)
	{
		for (auto i = 0u; i < generator->getNumTriangleMeshes(); i++)
		{
			auto it1 = configuration.triangleMeshes.find(i);
			if (it1 == configuration.triangleMeshes.end())
				throw std::runtime_error("it1 == configuration.triangleMeshes.end()");
			auto it2 = materials.find(it1->second.materialRef);
			if (it2 != materials.end())
				it2->second->activate(0);
			generator->getTriangleMesh(i)->draw();
			if (it2 != materials.end())
				it2->second->deactivate();
		}
		for (auto i = 0u; i < generator->getNumInstancedTriangleMeshes(); i++)
		{
			auto it1 = configuration.instancedTriangleMeshes.find(i);
			if (it1 == configuration.instancedTriangleMeshes.end())
				throw std::runtime_error("it1 == configuration.instancedTriangleMeshes.end()");
			auto it2 = materials.find(it1->second.materialRef);
			if (it2 != materials.end())
				it2->second->activate(0);
			generator->getInstancedTriangleMesh(i)->draw();
			if (it2 != materials.end())
				it2->second->deactivate();
		}
	}

	SwapBuffers(deviceContextHandle);
}

//////////////////////////////////////////////////////////////////////////
void TestRunnerApp::processKeys(float deltaTime)
{
	if (pressedKeys[VK_ESCAPE])
		DestroyWindow(windowHandle);
	if (keys[VK_LEFT] || keys[65]) // letter A
		moveCameraLeft(deltaTime);
	if (keys[VK_RIGHT] || keys[68]) // letter D
		moveCameraRight(deltaTime);
	if (keys[VK_UP] || keys[87]) // letter W
		moveCameraForward(deltaTime);
	if (keys[VK_DOWN] || keys[83]) // letter S
		moveCameraBackward(deltaTime);
	if (keys[81] || keys[33])  // letter Q
		moveCameraUp(deltaTime);
	if (keys[69] || keys[34]) // letter E
		moveCameraDown(deltaTime);
	if (pressedKeys[VK_F2])
		exportToObj();
	if (pressedKeys[VK_F4])
		generatePGASeeds();
	if (pressedKeys[VK_F5])
		generateGeometry();
	if (pressedKeys[VK_F6])
		generateGeometryEveryFrame = !generateGeometryEveryFrame;
	if (pressedKeys[VK_F7])
		generatePGASeedsEveryFrame = !generatePGASeedsEveryFrame;
	if (pressedKeys[VK_F8])
		outputGeometryBuffersUsage();
	if (pressedKeys[77]) // letter M
	{
		maximizeNumElements();
		generateGeometry();
	}
	if (pressedKeys[VK_ADD]) // plus
	{
		incrementNumElements();
		generateGeometry();
	}
	if (pressedKeys[VK_SUBTRACT]) // minus
	{
		decrementNumElements();
		generateGeometry();
	}
	if (pressedKeys[49]) // num 1
	{
		setAttributeIndex(1);
		generateGeometry();
	}
	if (pressedKeys[50]) // num 2
	{
		setAttributeIndex(2);
		generateGeometry();
	}
	if (pressedKeys[51]) // num 3
	{
		setAttributeIndex(3);
		generateGeometry();
	}
	if (pressedKeys[52]) // num 4
	{
		setAttributeIndex(4);
		generateGeometry();
	}
	if (pressedKeys[53]) // num 5
	{
		setAttributeIndex(5);
		generateGeometry();
	}
	if (pressedKeys[54]) // num 6
	{
		setAttributeIndex(6);
		generateGeometry();
	}
	if (pressedKeys[55]) // num 7
	{
		setAttributeIndex(7);
		generateGeometry();
	}
	if (pressedKeys[56]) // num 8
	{
		setAttributeIndex(8);
		generateGeometry();
	}
	if (pressedKeys[57]) // num 9
	{
		setAttributeIndex(9);
		generateGeometry();
	}
	if (pressedKeys[58]) // num 0
	{
		setAttributeIndex(0);
		generateGeometry();
	}
	if (pressedKeys[VK_PRIOR]) // page up
	{
		incrementAttribute();
		generateGeometry();
	}
	if (pressedKeys[VK_NEXT]) // page down
	{
		decrementAttribute();
		generateGeometry();
	}
}

//////////////////////////////////////////////////////////////////////////
void TestRunnerApp::keyDown(unsigned int virtualKey)
{
	keys[virtualKey] = true;
	pressedKeys[virtualKey] = true;
}

//////////////////////////////////////////////////////////////////////////
void TestRunnerApp::keyUp(unsigned int virtualKey)
{
	keys[virtualKey] = false;
}

//////////////////////////////////////////////////////////////////////////
void TestRunnerApp::mouseButtonDown(unsigned int button, int x, int y)
{
	if (button == MK_LBUTTON)
	{
		lastMousePosition = math::float2((float)x, (float)y);
		mouseButtonPressed = true;
	}
}

//////////////////////////////////////////////////////////////////////////
void TestRunnerApp::mouseButtonUp(unsigned int button, int x, int y)
{
	if (button == MK_LBUTTON)
	{
		mouseButtonPressed = false;
	}
}

//////////////////////////////////////////////////////////////////////////
void TestRunnerApp::mouseMove(int x, int y)
{
	if (!mouseButtonPressed)
		return;

	if (lastMousePosition.x == x && lastMousePosition.y == y)
		return;

	math::float2 mousePosition((float)x, (float)y);

	if (lastMousePosition.x < 0 && lastMousePosition.y < 0)
	{
		lastMousePosition = mousePosition;
		return;
	}

	math::float2 mouseDirection = normalize(mousePosition - lastMousePosition);
	if (mouseDirection.x > 0)
		turnCameraRight();
	else if (mouseDirection.x < 0)
		turnCameraLeft();

	if (mouseDirection.y > 0)
		turnCameraUp();
	else if (mouseDirection.y < 0)
		turnCameraDown();

	lastMousePosition = mousePosition;
}

//////////////////////////////////////////////////////////////////////////
void TestRunnerApp::turnCameraUp()
{
	cameraTheta = math::clamp(cameraTheta + ANGLE_INCREMENT, -CAMERA_PITCH_LIMIT, CAMERA_PITCH_LIMIT);
	updateCameraRotation();
}

//////////////////////////////////////////////////////////////////////////
void TestRunnerApp::turnCameraDown()
{
	cameraTheta = math::clamp(cameraTheta - ANGLE_INCREMENT, -CAMERA_PITCH_LIMIT, CAMERA_PITCH_LIMIT);
	updateCameraRotation();
}

//////////////////////////////////////////////////////////////////////////
void TestRunnerApp::turnCameraLeft()
{
	cameraPhi += ANGLE_INCREMENT /* NOTE: handiness sensitive */;
	updateCameraRotation();
}

//////////////////////////////////////////////////////////////////////////
void TestRunnerApp::turnCameraRight()
{
	cameraPhi -= ANGLE_INCREMENT /* NOTE: handiness sensitive */;
	updateCameraRotation();
}

//////////////////////////////////////////////////////////////////////////
void TestRunnerApp::updateCameraRotation()
{
	float cp = math::cos(cameraPhi);
	float sp = math::sin(cameraPhi);
	float ct = math::cos(cameraTheta);
	float st = math::sin(cameraTheta);
	auto w = -math::float3(ct * cp, st, ct * sp); /* NOTE: handiness sensitive */;
	auto v = math::float3(-st * cp, ct, -st * sp); /* NOTE: handiness sensitive */;
	auto u = cross(v, w); /* NOTE: handiness sensitive */;
	camera.u() = u;
	camera.v() = v;
	camera.w() = w;
}

//////////////////////////////////////////////////////////////////////////
void TestRunnerApp::generatePGASeeds()
{
	auto numAxioms = getNumAxioms();
	std::vector<float> seeds(numAxioms);
	for (auto i = 0u; i < numAxioms; i++)
		seeds[i] = uniform(prng);
#if defined(PGA_CPU)
	PGA::Host::setSeeds(&seeds[0], numAxioms);
#else
	PGA::Device::setSeeds(&seeds[0], numAxioms);
#endif
}

//////////////////////////////////////////////////////////////////////////
void TestRunnerApp::createGenerator()
{
	generator = std::unique_ptr<PGA::Rendering::Generator>(new PGA::Rendering::Generator());
	configuration.loadFromString(getConfigurationString());
	generator->build(createTriangleMeshes(), createInstancedTriangleMeshes());
	PGA::Rendering::Device::setNumTriangleMeshes(static_cast<unsigned int>(configuration.triangleMeshes.size()));
	PGA::Rendering::Device::setNumInstancedTriangleMeshes(static_cast<unsigned int>(configuration.instancedTriangleMeshes.size()));
	std::vector<unsigned int> maxNumVertices;
	std::vector<unsigned int> maxNumIndices;
	maxNumVertices.resize(PGA::Rendering::Constants::MaxNumTriangleMeshes);
	maxNumIndices.resize(PGA::Rendering::Constants::MaxNumTriangleMeshes);
	for (auto& entry : configuration.triangleMeshes)
	{
		if (entry.first >= PGA::Rendering::Constants::MaxNumInstancedTriangleMeshes)
			throw std::runtime_error("more triangle meshes than allowed");
		auto& triangleMesh = entry.second;
		maxNumVertices[entry.first] = triangleMesh.maxNumVertices;
		maxNumIndices[entry.first] = triangleMesh.maxNumIndices;
	}
}

//////////////////////////////////////////////////////////////////////////
void TestRunnerApp::outputGeometryBuffersUsage()
{
	std::cout << "Geometry buffer usage: " << std::endl;
	for (auto i = 0; i < generator->getNumTriangleMeshes(); i++)
	{
		const auto& triangleMesh = generator->getTriangleMesh(i);
		std::cout << "\ttriangle mesh " << i << ": " << 
			"(max. vert./curr. vert):  " << triangleMesh->getMaxNumVertices() << " / " << triangleMesh->getNumVertices() <<
			", " <<
			"(max. ind./curr. ind): " << triangleMesh->getMaxNumIndices() << " / " << triangleMesh->getNumIndices() <<
			std::endl;
	}
	for (auto i = 0; i < generator->getNumInstancedTriangleMeshes(); i++)
	{
		const auto& instancedTriangleMesh = generator->getInstancedTriangleMesh(i);
		std::cout << "\tinstanced triangle mesh " << i << ": " <<
			instancedTriangleMesh->getMaxNumInstances() << " / " <<
			instancedTriangleMesh->getNumInstances() << std::endl;
	}
}


//////////////////////////////////////////////////////////////////////////
PGA::Rendering::OBJExporter::Material TestRunnerApp::createOBJMaterial(const std::string& name, const std::string& baseFilePath, PGA::Rendering::Configuration::Material& material) const
{
	PGA::Rendering::OBJExporter::Material objMaterial;
	objMaterial.name = name;
	if (material.hasAttribute("color0"))
		objMaterial.diffuse = material.getFloat4Attribute("color0").xyz();
	if (material.hasAttribute("tex0"))
	{
		std::string textureFilename = baseFilePath + "_" + name + ".png";
		auto image = PGA::Rendering::PNG::loadFromDisk(configuration.textureRootPath + material.getAttribute("tex0"));
		PGA::Rendering::PNG::writeToDisk(textureFilename, rawData(image), width(image), height(image));
		objMaterial.diffuseMap = textureFilename;
	}
	if (material.hasAttribute("useUvX0"))
		objMaterial.useUvX = (material.getFloatAttribute("useUvX0") != 0);
	if (material.hasAttribute("useUvY0"))
		objMaterial.useUvY = (material.getFloatAttribute("useUvY0") != 0);
	if (material.hasAttribute("uvScale0"))
		objMaterial.uvScale = material.getFloat2Attribute("uvScale0");
	return objMaterial;
}

//////////////////////////////////////////////////////////////////////////
void TestRunnerApp::exportToObj()
{
	std::ofstream objFile("generated_geometry.obj");
	std::string mtlFileName = "generated_geometry.mtl";
	std::ofstream mtlFile(mtlFileName);

	std::map<unsigned int, PGA::Rendering::OBJExporter::Material> materials;
	for (auto& entry : configuration.triangleMeshes)
	{
		auto it = configuration.materials.find(entry.second.materialRef);
		if (it == configuration.materials.end())
			continue;
		materials.emplace(entry.first, createOBJMaterial("material_1_" + std::to_string(entry.first), "generated_geometry", it->second));
	}
	auto offset = static_cast<unsigned int>(configuration.triangleMeshes.size());
	for (auto& entry : configuration.instancedTriangleMeshes)
	{
		auto it = configuration.materials.find(entry.second.materialRef);
		if (it == configuration.materials.end())
			continue;
		materials.emplace(offset + entry.first, createOBJMaterial("material_1_" + std::to_string(entry.first), "generated_geometry", it->second));
	}

	// FIXME:
	generator->exportToOBJ(
		objFile,
		mtlFile,
		"generated_geometry",
		mtlFileName,
		materials,
		false
		);

	objFile.close();
	mtlFile.close();
}

//////////////////////////////////////////////////////////////////////////
LRESULT CALLBACK WinProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
	int x, y;

	switch (message)
	{
	case WM_DESTROY:
		PostQuitMessage(0);
		break;
	case WM_KEYDOWN:
		TestRunnerApp::s_instance->keyDown((unsigned int)wParam);
		break;
	case WM_KEYUP:
		TestRunnerApp::s_instance->keyUp((unsigned int)wParam);
		break;
	case WM_LBUTTONDOWN:
		x = GET_X_LPARAM(lParam);
		y = GET_Y_LPARAM(lParam);

		TestRunnerApp::s_instance->mouseButtonDown(MK_LBUTTON, x, y);
		break;
	case WM_MBUTTONDOWN:
		x = GET_X_LPARAM(lParam);
		y = GET_Y_LPARAM(lParam);

		TestRunnerApp::s_instance->mouseButtonDown(MK_MBUTTON, x, y);
		break;
	case WM_RBUTTONDOWN:
		x = GET_X_LPARAM(lParam);
		y = GET_Y_LPARAM(lParam);

		TestRunnerApp::s_instance->mouseButtonDown(MK_RBUTTON, x, y);
		break;
	case WM_LBUTTONUP:
		x = GET_X_LPARAM(lParam);
		y = GET_Y_LPARAM(lParam);

		TestRunnerApp::s_instance->mouseButtonUp(MK_LBUTTON, x, y);
		break;
	case WM_MBUTTONUP:
		x = GET_X_LPARAM(lParam);
		y = GET_Y_LPARAM(lParam);

		TestRunnerApp::s_instance->mouseButtonUp(MK_MBUTTON, x, y);
		break;
	case WM_RBUTTONUP:
		x = GET_X_LPARAM(lParam);
		y = GET_Y_LPARAM(lParam);

		TestRunnerApp::s_instance->mouseButtonUp(MK_RBUTTON, x, y);
		break;
	case WM_MOUSEMOVE:
		x = GET_X_LPARAM(lParam);
		y = GET_Y_LPARAM(lParam);

		TestRunnerApp::s_instance->mouseMove(x, y);
		break;
	default:
		return DefWindowProc(hWnd, message, wParam, lParam);
		break;
	}

	return 0;
}
