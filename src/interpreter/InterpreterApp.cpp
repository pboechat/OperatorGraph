//#define VERBOSE

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
#include <pga/core/StringUtils.h>
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
#include <pga/compiler/Parser.h>
#include <pga/compiler/DispatchTableGenerator.h>
#include <pga/compiler/Context.h>
#include <pga/compiler/Graph.h>
#include <pga/compiler/CodeGenerator.h>

#include "Constants.h"
#include "PGAFacade.h"
#define INTERPRETER_EXTERN
#include "GlobalVariables.cuh"
#include "InterpreterApp.h"

//////////////////////////////////////////////////////////////////////////
#define win32Assert(resultHandle, errorMessage) \
	if (resultHandle == 0) \
		{ \
			std::cerr << ##errorMessage << std::endl; \
			dispose(); \
			exit(EXIT_FAILURE); \
		} \

//////////////////////////////////////////////////////////////////////////
class GeneratorBufferOverflow : public std::exception {};

//////////////////////////////////////////////////////////////////////////
struct CppSourceCodeCallback : PGA::Compiler::Graph::ComputePartitionCallback
{
	virtual bool operator()(std::size_t i, PGA::Compiler::Graph::PartitionPtr& partition)
	{
		std::stringstream out;
		PGA::Compiler::CodeGenerator::fromPartition(out, partition);
		std::cout << "[CPP SOURCE]" << std::endl << out.str() << std::endl << std::endl;
		return true;
	}

};

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
InterpreterApp* InterpreterApp::s_instance = 0;
const char* InterpreterApp::WINDOW_TITLE = "interpreter";
const char* InterpreterApp::WINDOW_CLASS_NAME = "interpreter_window";
const unsigned int InterpreterApp::SCREEN_WIDTH = 1024;
const unsigned int InterpreterApp::SCREEN_HEIGHT = 768;
const unsigned int InterpreterApp::BYTES_PER_PIXEL = 4;
const unsigned int InterpreterApp::COLOR_BUFFER_BITS = 32;
const unsigned int InterpreterApp::DEPTH_BUFFER_BITS = 32;
const unsigned int InterpreterApp::HAS_ALPHA = 0;
const PIXELFORMATDESCRIPTOR InterpreterApp::PIXEL_FORMAT_DESCRIPTOR = { sizeof(PIXELFORMATDESCRIPTOR), 1, PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER, PFD_TYPE_RGBA, COLOR_BUFFER_BITS, 0, 0, 0, 0, 0, 0, HAS_ALPHA, 0, 0, 0, 0, 0, 0, DEPTH_BUFFER_BITS, 0, 0, PFD_MAIN_PLANE, 0, 0, 0, 0 };
const float InterpreterApp::ANGLE_INCREMENT = 0.05f;
const float InterpreterApp::CAMERA_PITCH_LIMIT = 1.0472f; // 60 deg.
const float InterpreterApp::CAMERA_MOVE_SPEED = 100.0f;

//////////////////////////////////////////////////////////////////////////
InterpreterApp::InterpreterApp() :
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
	interpretEveryFrame(false),
	configurationFilename("interpreter_configuration.xml"),
	dAxioms(0)
{
	memset(keys, 0, sizeof(bool) * 0xff);
	memset(pressedKeys, 0, sizeof(bool) * 0xff);
	s_instance = this;
}

//////////////////////////////////////////////////////////////////////////
InterpreterApp::~InterpreterApp()
{
	s_instance = nullptr;
}

//////////////////////////////////////////////////////////////////////////
void InterpreterApp::freeAxioms()
{
	if (dAxioms)
	{
#if defined(PGA_CPU)
		free(dAxioms);
#else
		PGA_CUDA_checkedCall(cudaFree(dAxioms));
#endif
		dAxioms = 0;
	}
}

//////////////////////////////////////////////////////////////////////////
void InterpreterApp::copyAxioms(std::vector<PGA::Compiler::Axiom>& compilerAxioms)
{
	if (compilerAxioms.empty())
		throw std::runtime_error("no starting axiom");
	if (compilerAxioms.size() > ::Constants::MaxNumAxioms)
		throw std::runtime_error("using more axioms than allowed");
	std::vector<Axiom> axioms;
	for (auto& compilerAxiom : compilerAxioms)
	{
		Axiom axiom;
		axiom.shapeType = compilerAxiom.shapeType;
		axiom.entryIndex = static_cast<int>(compilerAxiom.entryIndex);
		axiom.numVertices = static_cast<unsigned int>(compilerAxiom.vertices.size());
		if (axiom.shapeType == PGA::Compiler::ShapeType::DYNAMIC_CONVEX_POLYGON ||
			axiom.shapeType == PGA::Compiler::ShapeType::DYNAMIC_POLYGON ||
			axiom.shapeType == PGA::Compiler::ShapeType::DYNAMIC_CONVEX_RIGHT_PRISM ||
			axiom.shapeType == PGA::Compiler::ShapeType::DYNAMIC_RIGHT_PRISM)
		{
			if (compilerAxiom.vertices.size() < 3)
				throw std::runtime_error("axiom is a dynamic polygon/polyhedra and uses less than 3 vertices");
			if (compilerAxiom.vertices.size() > PGA::Constants::MaxNumSides)
				throw std::runtime_error("axiom uses more vertices than allowed");
			// NOTE: orderVertices_CW(..) works only with convex polygons
			if (axiom.shapeType == PGA::Compiler::ShapeType::DYNAMIC_CONVEX_POLYGON ||
				axiom.shapeType == PGA::Compiler::ShapeType::DYNAMIC_CONVEX_RIGHT_PRISM)
			{
				std::vector<math::float2> orderedVertices;
				PGA::GeometryUtils::orderVertices_CCW(compilerAxiom.vertices, orderedVertices);
				for (unsigned int j = 0; j < axiom.numVertices; j++)
					axiom.vertices[j] = orderedVertices[j];
			}
			else
			{
				std::vector<math::float2> nonCollinearVertices;
				PGA::GeometryUtils::removeCollinearPoints(compilerAxiom.vertices, nonCollinearVertices);
				axiom.numVertices = static_cast<unsigned int>(nonCollinearVertices.size());
				for (auto j = 0; j < nonCollinearVertices.size(); j++)
					axiom.vertices[j] = nonCollinearVertices[j];

			}
		}
		axioms.emplace_back(axiom);
	}
#ifdef PGA_CPU
	dAxioms = Host::setAxioms(static_cast<unsigned int>(axioms.size()), &axioms[0]);
#else
	dAxioms = Device::setAxioms(static_cast<unsigned int>(axioms.size()), &axioms[0]);
#endif
}

//////////////////////////////////////////////////////////////////////////
void InterpreterApp::parse()
{
	freeAxioms();

	std::ifstream in(pgaSourceFilename);
	std::string sourceCode((std::istreambuf_iterator<char>(in)),
		std::istreambuf_iterator<char>());
	std::vector<PGA::Compiler::Axiom> axioms;
	std::vector<PGA::Compiler::Rule> rules;
	PGA::Compiler::Logger logger;
	if (!PGA::Compiler::Parser::parse(sourceCode, logger, axioms, rules))
	{
		std::cout << "[PARSER WARNINGS]: " << std::endl << logger[PGA::Compiler::Logger::LL_WARNING].str() << std::endl;
		std::cerr << "[PARSER ERRORS]: " << std::endl << logger[PGA::Compiler::Logger::LL_ERROR].str() << std::endl;
		throw std::runtime_error("parsing error");
	}

	std::set<std::size_t> alwaysCutEdges;
	PGA::Compiler::DispatchTableGenerator generator;
	PGA::Compiler::Context context(axioms[0], rules);

	context.baseGraph.computePhases(alwaysCutEdges);

	if (!generator.fromBaseGraph(context.baseGraph, getProcedureList(), logger))
	{
		std::cout << "[PARSER WARNINGS]: " << std::endl << logger[PGA::Compiler::Logger::LL_WARNING].str() << std::endl;
		std::cout << "[PARSER ERRORS]: " << std::endl << logger[PGA::Compiler::Logger::LL_ERROR].str() << std::endl;
		throw std::runtime_error("code generator error");
	}

	std::cout << "[PARSER WARNINGS]: " << std::endl << logger[PGA::Compiler::Logger::LL_WARNING].str() << std::endl << std::endl;

	// FIXME: no symbol to entryIndex in operator graph version of dpt generator yet
	axioms[0].entryIndex = 0;

	std::cout << "[TERMINAL SYMBOLS]: " << std::endl;;
	for (auto& terminalSymbol : context.terminalSymbols)
		std::cout << terminalSymbol << std::endl;
	std::cout << std::endl;

	std::string partitionUid;
	for (auto i = 0; i < context.baseGraph.numEdges(); i++)
		partitionUid += "1";

	context.baseGraph.computePartition(
		CppSourceCodeCallback(),
		true,
		PGA::Compiler::Graph::Partition::cutEdgesFromUid(partitionUid)
	);

	initializePGA(generator.dispatchTable);
	copyAxioms(axioms);

	std::vector<float> seeds(axioms.size());
	for (auto i = 0u; i < axioms.size(); i++)
		seeds[i] = uniform(prng);
#if defined(PGA_CPU)
	PGA::Host::setSeeds(&seeds[0], static_cast<unsigned int>(axioms.size()));
#else
	PGA::Device::setSeeds(&seeds[0], static_cast<unsigned int>(axioms.size()));
#endif
}

//////////////////////////////////////////////////////////////////////////
void InterpreterApp::interpret()
{
	parse();
	if (generator == nullptr)
		createGenerator();
	generator->bind();
	executePGA();
	if (hasBufferOverflow())
		throw GeneratorBufferOverflow();
	generator->unbind();
}

//////////////////////////////////////////////////////////////////////////
void InterpreterApp::createMaterial(int materialRef)
{
	auto it = configuration.materials.find(materialRef);
	if (it == configuration.materials.end())
		return;
#ifdef VERBOSE
	if (materialRef < 0)
	{
		std::cout << "\t************************************************************" << std::endl;
		std::cout << "\tNested material" << std::endl;
		std::cout << "\t************************************************************" << std::endl;
	}
	else
	{
		std::cout << "\t************************************************************" << std::endl;
		std::cout << "\tMaterial " << materialRef << std::endl;
		std::cout << "\t************************************************************" << std::endl;
	}
#endif
	auto& material = it->second;
	std::unique_ptr<PGA::Rendering::Material> newMaterial(new PGA::Rendering::Material());
#ifdef VERBOSE
	std::cout << "\ttype: " << material.type << std::endl;
	std::cout << "\tattributes: " << std::endl;
#endif
	switch (material.type)
	{
	case 0: // Color
		newMaterial->setShader(
			std::unique_ptr<PGA::Rendering::ColorShader>(new PGA::Rendering::ColorShader())
			);
#ifdef VERBOSE
#endif
		if (material.hasAttribute("color0"))
		{
			auto color0 = material.getFloat4Attribute("color0");
#ifdef VERBOSE
			std::cout << "\t\tcolor0: (" << color0.x << ", " << color0.y << ", " << color0.z << ", " << color0.w << ")" << std::endl;
#endif
			newMaterial->setParameter("color0", color0);
		}
		else
			std::cout << "[WARNING] material parameter color0 not set [materialRef=" << std::to_string(materialRef) << "]" << std::endl;
		break;
	case 1: // Textured
		newMaterial->setShader(
			std::unique_ptr<PGA::Rendering::TexturedShader>(new PGA::Rendering::TexturedShader())
			);
		if (material.hasAttribute("tex0"))
		{
#ifdef VERBOSE
			std::cout << "\ttex0: ";
#endif
			auto textureFilename = configuration.textureRootPath + material.getAttribute("tex0");
#ifdef VERBOSE
			std::cout << textureFilename << std::endl;
#endif
			auto texture = loadTexture2D(textureFilename);
			if (texture != nullptr)
				newMaterial->setParameter("tex0", std::move(texture));
			else
				std::cout << "[WARNING] could not load texture [materialRef=" << std::to_string(materialRef) << ", textureFilename=\"" << textureFilename << "\"]" << std::endl;
		}
		else
			std::cout << "[WARNING] material parameter tex0 not set [materialRef=" << std::to_string(materialRef) << "]" << std::endl;
		if (material.hasAttribute("useUvX0"))
		{
			auto useUvX0 = material.getFloatAttribute("useUvX0");
#ifdef VERBOSE
			std::cout << "\t\tuseUvX0: " << useUvX0 << std::endl;
#endif
			newMaterial->setParameter("useUvX0", useUvX0);
		}
		else
			newMaterial->setParameter("useUvX0", 1.0f);
		if (material.hasAttribute("useUvY0"))
		{
			auto useUvY0 = material.getFloatAttribute("useUvY0");
#ifdef VERBOSE
			std::cout << "\t\tuseUvY0: " << useUvY0 << std::endl;
#endif
			newMaterial->setParameter("useUvY0", useUvY0);
		}
		else
			newMaterial->setParameter("useUvY0", 1.0f);
		if (material.hasAttribute("uvScale0"))
		{
			auto uvScale0 = material.getFloat2Attribute("uvScale0");
#ifdef VERBOSE
			std::cout << "\t\tuvScale0: (" << uvScale0.x << ", " << uvScale0.y << ")" << std::endl;
#endif
			newMaterial->setParameter("uvScale0", uvScale0);
		}
		else
			newMaterial->setParameter("uvScale0", math::float2(1.0f, 1.0f));
		break;
	default:
		throw std::runtime_error("unknown material type [materialRef=" + std::to_string(materialRef) + ", configMaterial.type=" + std::to_string(material.type) + "]");
	}
#ifdef VERBOSE
	std::cout << "\tbackFaceCulling: " << material.backFaceCulling << std::endl;
#endif
	newMaterial->setBackFaceCulling(material.backFaceCulling);
	materials.emplace(materialRef, std::move(newMaterial));
}

//////////////////////////////////////////////////////////////////////////
std::vector<std::unique_ptr<PGA::Rendering::TriangleMesh>> InterpreterApp::createTriangleMeshes()
{
	std::vector<std::unique_ptr<PGA::Rendering::TriangleMesh>> triangleMeshes;
	std::string textureFilename;
	std::unique_ptr<PGA::Rendering::Texture> texture;
	for (auto& entry : configuration.triangleMeshes)
	{
#ifdef VERBOSE
		std::cout << "************************************************************" << std::endl;
		std::cout << "Triangle mesh " << triangleMeshes.size() << std::endl;
		std::cout << "************************************************************" << std::endl;
#endif
		auto& triangleMesh = entry.second;
		createMaterial(triangleMesh.materialRef);
#ifdef VERBOSE
		std::cout << "maxNumVertices: " << triangleMesh.maxNumVertices << std::endl;
		std::cout << "maxNumIndices: " << triangleMesh.maxNumIndices << std::endl;
#endif
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
std::vector<std::unique_ptr<PGA::Rendering::InstancedTriangleMesh>> InterpreterApp::createInstancedTriangleMeshes()
{
	std::vector<std::unique_ptr<PGA::Rendering::InstancedTriangleMesh>> instancedTriangleMeshes;
	for (auto& entry : configuration.instancedTriangleMeshes)
	{
#ifdef VERBOSE
		std::cout << "************************************************************" << std::endl;
		std::cout << "Instanced triangle mesh " << instancedTriangleMeshes.size() << std::endl;
		std::cout << "************************************************************" << std::endl;
#endif
		auto& instancedTriangleMesh = entry.second;
		createMaterial(instancedTriangleMesh.materialRef);
		std::unique_ptr<PGA::Rendering::InstancedTriangleMeshSource> source;
#ifdef VERBOSE
		std::cout << "type: " << instancedTriangleMesh.type << std::endl;
#endif
		switch (instancedTriangleMesh.type)
		{
		case PGA::Rendering::Configuration::InstancedTriangleMesh::SHAPE:
#ifdef VERBOSE
			std::cout << "shape: " << instancedTriangleMesh.shape << std::endl;
#endif
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
		{
			std::string modelPath = configuration.modelRootPath + instancedTriangleMesh.modelPath;
#ifdef VERBOSE
			std::cout << "model: " << modelPath << std::endl;
#endif
			source = std::unique_ptr<PGA::Rendering::OBJMesh>(
				new PGA::Rendering::OBJMesh(modelPath)
			);
			break;
		}
		default:
			throw std::runtime_error("unknown instanced triangle mesh type");
		}
#ifdef VERBOSE
		std::cout << "maxNumElements: " << instancedTriangleMesh.maxNumElements << std::endl;
#endif
		instancedTriangleMeshes.emplace_back(new PGA::Rendering::InstancedTriangleMesh(
				instancedTriangleMesh.maxNumElements,
				std::move(source)
			)
		);
	}
	return instancedTriangleMeshes;
}

//////////////////////////////////////////////////////////////////////////
bool InterpreterApp::hasBufferOverflow()
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
void InterpreterApp::finalizePGAWithException()
{
	for (auto& material : materials)
		material.second.release();
	generator.release();
	releasePGA();
}

//////////////////////////////////////////////////////////////////////////
int InterpreterApp::run(unsigned int argc, const char** argv)
{
	//////////////////////////////////////////////////////////////////////////
	// Read command line arguments

	int deviceIndex = 0;
	long globalSeed = -1;
	for (auto i = 0u; i < argc; i++)
	{
		std::vector<std::string> tokens;
		PGA::StringUtils::split(argv[i], '=', tokens);
		if (tokens.size() < 2)
			continue;
		std::string key = tokens[0], value = tokens[1];
		if (key == "src")
			pgaSourceFilename = value;
		else if (key == "config")
			configurationFilename = value;
		else if (key == "device")
			deviceIndex = atoi(value.c_str());
		else if (key == "seed")
			globalSeed = atol(value.c_str());
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
		exit(EXIT_FAILURE);
	}

	int returnValue = EXIT_SUCCESS;
	try
	{
		//////////////////////////////////////////////////////////////////////////
		// One-time initialization

		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, deviceIndex);
		cudaGLSetGLDevice(deviceIndex);

		std::cout << "device name: " << deviceProp.name << std::endl;
		std::cout << "pga source file name: " << pgaSourceFilename << std::endl << std::endl;

		PGA::Rendering::Shader::setIncludePath({ "shaders/" });

		
		glEnable(GL_DEPTH_TEST);
		glViewport(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT);

		glGenBuffers(1, &cameraUniformBuffer);

		//////////////////////////////////////////////////////////////////////////
		// Main application loop

		static bool firstTime = true;
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
				auto start = std::chrono::system_clock::now().time_since_epoch();
				if (firstTime || interpretEveryFrame)
				{
					interpret();
					if (firstTime)
						firstTime = false;
				}
				drawScene();
				auto end = std::chrono::system_clock::now().time_since_epoch();
				float deltaTime = (std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000000.0f);
				processKeys(deltaTime);
				memset(pressedKeys, 0, sizeof(bool) * 0xFF);
				std::stringstream stream;
				stream << std::fixed << std::setprecision(5) << WINDOW_TITLE << " @ fps: " << (1 / deltaTime);
				SetWindowText(windowHandle, stream.str().c_str());
			}
		}
		returnValue = EXIT_SUCCESS;

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
		returnValue = EXIT_FAILURE;
		finalizePGAWithException();
	}
	catch (PGA::Rendering::GL::Exception& e)
	{
		std::cerr << e.what() << std::endl;
		DestroyWindow(windowHandle);
		returnValue = EXIT_FAILURE;
		finalizePGAWithException();
	}
	catch (GeneratorBufferOverflow&)
	{
		std::cerr << "generator buffer overflow" << std::endl;
		DestroyWindow(windowHandle);
		returnValue = EXIT_FAILURE;
		finalizePGAWithException();
	}
	catch (std::runtime_error& e)
	{
		std::cerr << e.what() << std::endl;
		DestroyWindow(windowHandle);
		returnValue = EXIT_FAILURE;
		finalizePGAWithException();
	}
	catch (...)
	{
		std::cerr << "unknown error" << std::endl;
		DestroyWindow(windowHandle);
		returnValue = EXIT_FAILURE;
		finalizePGAWithException();
	}

	dispose();

	DestroyWindow(windowHandle);
	win32Assert(UnregisterClass(WINDOW_CLASS_NAME, applicationHandle), "UnregisterClass() failed");

	deviceContextHandle = 0;
	windowHandle = 0;
	applicationHandle = 0;

	return returnValue;
}

//////////////////////////////////////////////////////////////////////////
void InterpreterApp::dispose()
{
	if (openGLRenderingContextHandle)
	{
		win32Assert(wglMakeCurrent(0, 0), "wglMakeCurrent() failed");
		win32Assert(wglDeleteContext(openGLRenderingContextHandle), "wglDeleteContext() failed");
		openGLRenderingContextHandle = 0;
	}
}

//////////////////////////////////////////////////////////////////////////
void InterpreterApp::moveCameraLeft(float deltaTime)
{
	camera.position() -= camera.u() * deltaTime * CAMERA_MOVE_SPEED;
}

//////////////////////////////////////////////////////////////////////////
void InterpreterApp::moveCameraRight(float deltaTime)
{
	camera.position() += camera.u() * deltaTime * CAMERA_MOVE_SPEED;
}

//////////////////////////////////////////////////////////////////////////
void InterpreterApp::moveCameraForward(float deltaTime)
{
	camera.position() += camera.w() * deltaTime * CAMERA_MOVE_SPEED;
}

//////////////////////////////////////////////////////////////////////////
void InterpreterApp::moveCameraBackward(float deltaTime)
{
	camera.position() -= camera.w() * deltaTime * CAMERA_MOVE_SPEED;
}

//////////////////////////////////////////////////////////////////////////
void InterpreterApp::moveCameraUp(float deltaTime)
{
	camera.position() += camera.v() * deltaTime * CAMERA_MOVE_SPEED;
}

//////////////////////////////////////////////////////////////////////////
void InterpreterApp::moveCameraDown(float deltaTime)
{
	camera.position() -= camera.v() * deltaTime * CAMERA_MOVE_SPEED;
}

//////////////////////////////////////////////////////////////////////////
void InterpreterApp::drawScene()
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
void InterpreterApp::processKeys(float deltaTime)
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
	if (pressedKeys[VK_F5])
		interpret();
	if (pressedKeys[VK_F6])
		interpretEveryFrame = !interpretEveryFrame;
	if (pressedKeys[VK_F7])
		generatePGASeedsEveryFrame = !generatePGASeedsEveryFrame;
	if (pressedKeys[VK_F8])
		outputGeometryBuffersUsage();
}

//////////////////////////////////////////////////////////////////////////
void InterpreterApp::keyDown(unsigned int virtualKey)
{
	keys[virtualKey] = true;
	pressedKeys[virtualKey] = true;
}

//////////////////////////////////////////////////////////////////////////
void InterpreterApp::keyUp(unsigned int virtualKey)
{
	keys[virtualKey] = false;
}

//////////////////////////////////////////////////////////////////////////
void InterpreterApp::mouseButtonDown(unsigned int button, int x, int y)
{
	if (button == MK_LBUTTON)
	{
		lastMousePosition = math::float2((float)x, (float)y);
		mouseButtonPressed = true;
	}
}

//////////////////////////////////////////////////////////////////////////
void InterpreterApp::mouseButtonUp(unsigned int button, int x, int y)
{
	if (button == MK_LBUTTON)
	{
		mouseButtonPressed = false;
	}
}

//////////////////////////////////////////////////////////////////////////
void InterpreterApp::mouseMove(int x, int y)
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
void InterpreterApp::turnCameraUp()
{
	cameraTheta = math::clamp(cameraTheta + ANGLE_INCREMENT, -CAMERA_PITCH_LIMIT, CAMERA_PITCH_LIMIT);
	updateCameraRotation();
}

//////////////////////////////////////////////////////////////////////////
void InterpreterApp::turnCameraDown()
{
	cameraTheta = math::clamp(cameraTheta - ANGLE_INCREMENT, -CAMERA_PITCH_LIMIT, CAMERA_PITCH_LIMIT);
	updateCameraRotation();
}

//////////////////////////////////////////////////////////////////////////
void InterpreterApp::turnCameraLeft()
{
	cameraPhi += ANGLE_INCREMENT /* NOTE: handiness sensitive */;
	updateCameraRotation();
}

//////////////////////////////////////////////////////////////////////////
void InterpreterApp::turnCameraRight()
{
	cameraPhi -= ANGLE_INCREMENT /* NOTE: handiness sensitive */;
	updateCameraRotation();
}

//////////////////////////////////////////////////////////////////////////
void InterpreterApp::updateCameraRotation()
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
void InterpreterApp::createGenerator()
{
	generator = std::unique_ptr<PGA::Rendering::Generator>(new PGA::Rendering::Generator());
	std::ifstream in(configurationFilename);
	std::string configurationString((std::istreambuf_iterator<char>(in)),
		std::istreambuf_iterator<char>());
	configuration.loadFromString(configurationString);
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
void InterpreterApp::outputGeometryBuffersUsage()
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
PGA::Rendering::OBJExporter::Material InterpreterApp::createOBJMaterial(const std::string& name, const std::string& baseFilePath, PGA::Rendering::Configuration::Material& material) const
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
void InterpreterApp::exportToObj()
{
	auto count = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
	std::ofstream objFile("generated_geometry_" + std::to_string(count) + ".obj");
	std::string mtlFileName = "generated_geometry_" + std::to_string(count) + ".mtl";
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
		"generated_geometry_" + std::to_string(count),
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
		InterpreterApp::s_instance->keyDown((unsigned int)wParam);
		break;
	case WM_KEYUP:
		InterpreterApp::s_instance->keyUp((unsigned int)wParam);
		break;
	case WM_LBUTTONDOWN:
		x = GET_X_LPARAM(lParam);
		y = GET_Y_LPARAM(lParam);

		InterpreterApp::s_instance->mouseButtonDown(MK_LBUTTON, x, y);
		break;
	case WM_MBUTTONDOWN:
		x = GET_X_LPARAM(lParam);
		y = GET_Y_LPARAM(lParam);

		InterpreterApp::s_instance->mouseButtonDown(MK_MBUTTON, x, y);
		break;
	case WM_RBUTTONDOWN:
		x = GET_X_LPARAM(lParam);
		y = GET_Y_LPARAM(lParam);

		InterpreterApp::s_instance->mouseButtonDown(MK_RBUTTON, x, y);
		break;
	case WM_LBUTTONUP:
		x = GET_X_LPARAM(lParam);
		y = GET_Y_LPARAM(lParam);

		InterpreterApp::s_instance->mouseButtonUp(MK_LBUTTON, x, y);
		break;
	case WM_MBUTTONUP:
		x = GET_X_LPARAM(lParam);
		y = GET_Y_LPARAM(lParam);

		InterpreterApp::s_instance->mouseButtonUp(MK_MBUTTON, x, y);
		break;
	case WM_RBUTTONUP:
		x = GET_X_LPARAM(lParam);
		y = GET_Y_LPARAM(lParam);

		InterpreterApp::s_instance->mouseButtonUp(MK_RBUTTON, x, y);
		break;
	case WM_MOUSEMOVE:
		x = GET_X_LPARAM(lParam);
		y = GET_Y_LPARAM(lParam);

		InterpreterApp::s_instance->mouseMove(x, y);
		break;
	default:
		return DefWindowProc(hWnd, message, wParam, lParam);
		break;
	}

	return 0;
}
