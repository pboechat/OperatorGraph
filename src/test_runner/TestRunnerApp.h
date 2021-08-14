#pragma once

#include "Camera.h"
#include "LogManager.h"

#include <math/vector.h>
#include <pga/rendering/Configuration.h>
#include <pga/rendering/Generator.h>
#include <pga/rendering/InstancedTriangleMesh.h>
#include <pga/rendering/Material.h>
#include <pga/rendering/TriangleMesh.h>
#include <windows.h>

#include <chrono>
#include <memory>
#include <random>
#include <string>
#include <vector>

int main(unsigned int argc, const char** argv);

LRESULT CALLBACK WinProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam);

class TestRunnerApp
{
public:
	static const unsigned int SCREEN_WIDTH;
	static const unsigned int SCREEN_HEIGHT;
	static const unsigned int BYTES_PER_PIXEL;

	inline static TestRunnerApp* getInstance()
	{
		return s_instance;
	}

	int run(unsigned int argc, const char** argv);

	friend int main(unsigned int argc, const char** argv);
	friend LRESULT CALLBACK WinProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam);

private:
	static const char* WINDOW_TITLE;
	static const char* WINDOW_CLASS_NAME;
	static const unsigned int COLOR_BUFFER_BITS;
	static const unsigned int DEPTH_BUFFER_BITS;
	static const unsigned int HAS_ALPHA;
	static const PIXELFORMATDESCRIPTOR PIXEL_FORMAT_DESCRIPTOR;
	static const float ANGLE_INCREMENT;
	static const float CAMERA_PITCH_LIMIT;
	static const float CAMERA_MOVE_SPEED;

	static TestRunnerApp* s_instance;

	HINSTANCE applicationHandle;
	HWND windowHandle;
	HDC deviceContextHandle;
	int pixelFormat;
	HGLRC openGLRenderingContextHandle;
	bool mouseButtonPressed;
	bool keys[0xff];
	bool pressedKeys[0xff];
	Camera camera;
	unsigned int cameraUniformBuffer;
	math::float2 lastMousePosition;
	float cameraPhi;
	float cameraTheta;
	std::unique_ptr<PGA::Rendering::Generator> generator;
	std::map<int, std::unique_ptr<PGA::Rendering::Material>> materials;
	PGA::Rendering::Configuration configuration;
	bool generatePGASeedsEveryFrame;
	bool generateGeometryEveryFrame;
	bool silent;
	LogManager logManager;
	std::default_random_engine prng;
	std::uniform_real_distribution<float> uniform;

	TestRunnerApp();
	~TestRunnerApp();

	void dispose();
	void keyDown(unsigned int virtualKey);
	void keyUp(unsigned int virtualKey);
	void mouseButtonDown(unsigned int button, int x, int y);
	void mouseButtonUp(unsigned int button, int x, int y);
	void mouseMove(int x, int y);
	void moveCameraLeft(float deltaTime);
	void moveCameraRight(float deltaTime);
	void moveCameraForward(float deltaTime);
	void moveCameraBackward(float deltaTime);
	void moveCameraUp(float deltaTime);
	void moveCameraDown(float deltaTime);
	void turnCameraUp();
	void turnCameraDown();
	void turnCameraLeft();
	void turnCameraRight();
	void updateCameraRotation();
	void drawScene();
	void processKeys(float deltaTime);
	double generateGeometry();
	void generatePGASeeds();
	void createMaterial(int materialRef);
	std::vector<std::unique_ptr<PGA::Rendering::TriangleMesh>> createTriangleMeshes();
	std::vector<std::unique_ptr<PGA::Rendering::InstancedTriangleMesh>> createInstancedTriangleMeshes();
	bool hasBufferOverflow();
	void finalizePGAWithException();
	void createGenerator();
	void outputGeometryBuffersUsage();
	PGA::Rendering::OBJExporter::Material createOBJMaterial(const std::string& name, const std::string& baseFilePath, PGA::Rendering::Configuration::Material& material) const;
	void exportToObj();

};
