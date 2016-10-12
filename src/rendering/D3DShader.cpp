#include <string.h>
#include <fstream>
#include <vector>
#include <list>
#include <algorithm>
#include <D3Dcompiler.h>
#include <D3Dcommon.h>

#include <pga/rendering/D3DException.h>
#include <pga/rendering/D3DShader.h>

namespace PGA
{
	namespace Rendering
	{
		namespace D3D
		{
			//////////////////////////////////////////////////////////////////////////
			std::vector<std::string> g_includePaths;

			//////////////////////////////////////////////////////////////////////////
			bool openShaderFileStream(
				std::string& fileName, 
				std::ifstream& file,
				D3D_INCLUDE_TYPE includeType = D3D_INCLUDE_SYSTEM
			)
			{
				switch (includeType)
				{
				case D3D_INCLUDE_LOCAL:
				{
					file = std::ifstream(fileName);
					if (file.is_open())
						return true;
					break;
				}
				case D3D_INCLUDE_SYSTEM:
				{
					std::string baseFileName(fileName);
					for (auto& includeDir : g_includePaths)
					{
						fileName = includeDir + "\\" + baseFileName;
						file = std::ifstream(fileName);
						if (file.is_open())
							return true;
					}
					fileName = baseFileName;
					file = std::ifstream(fileName);
					if (file.is_open())
						return true;
					break;
				}
				default:
					throw std::runtime_error("PGA::Rendering::D3D::openShaderStream(..): unsupported include type [includeType=" + std::to_string(includeType) + "]");
				}
				return false;
			}

			//////////////////////////////////////////////////////////////////////////
			class ShaderIncludeManager : public ID3DInclude
			{
			private:
				std::list<std::unique_ptr<char[]>> data;

			public:
				HRESULT ShaderIncludeManager::Open(D3D_INCLUDE_TYPE IncludeType, LPCSTR pFileName, LPCVOID pParentData, LPCVOID* ppData, UINT* pBytes)
				{
					std::ifstream file;
					for (auto& includePath : g_includePaths)
					{
						file = std::ifstream(includePath + pFileName, std::ios::binary);
						if (file.is_open())
							break;
					}
					if (!file.is_open())
					{
						*ppData = NULL;
						*pBytes = 0;
						return S_OK;
					}
					file.seekg(0, std::ios::end);
					UINT size = static_cast<UINT>(file.tellg());
					file.seekg(0, std::ios::beg);
					std::unique_ptr<char[]> newData(new char[size]);
					file.read(newData.get(), size);
					*ppData = newData.get();
					*pBytes = size;
					data.push_back(std::move(newData));
					return S_OK;
				}

				HRESULT ShaderIncludeManager::Close(LPCVOID pData)
				{
					data.remove_if([pData](const std::unique_ptr<char[]>& ptr) { return reinterpret_cast<LPCVOID>(ptr.get()) == pData; });
					return S_OK;
				}

			};

			//////////////////////////////////////////////////////////////////////////
			std::unique_ptr<ShaderIncludeManager> g_shaderIncludeManager = nullptr;

			//////////////////////////////////////////////////////////////////////////
			ShaderIncludeManager* getShaderIncludeManager()
			{
				if (g_shaderIncludeManager == nullptr)
					g_shaderIncludeManager = std::make_unique<ShaderIncludeManager>();
				return g_shaderIncludeManager.get();
			}

			ID3D10Blob* compileShader(const std::string& fileName, const std::string& shaderModel)
			{
				std::string endFileName(fileName);
				std::ifstream in;
				if (!openShaderFileStream(endFileName, in))
					throw std::runtime_error("PGA::Rendering::D3D::compileShader(..): file not found [fileName=" + fileName + "]");
				std::string sourceCode((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
				ID3D10Blob* shaderBinary, *errorMessage;
				if (FAILED(D3DCompile(sourceCode.c_str(), sourceCode.length(), endFileName.c_str(), 0, getShaderIncludeManager(), "main", shaderModel.c_str(), 0, 0, &shaderBinary, &errorMessage)))
				{
					std::string compileErrors((char*)(errorMessage->GetBufferPointer()));
					errorMessage->Release();
					std::cout << compileErrors << std::endl;
					throw std::runtime_error("PGA::Rendering::D3D::compileShader(..): " + compileErrors);
				}
				return shaderBinary;
			}

			ID3D11VertexShader* createVertexShader(ID3D11Device* device, ID3D10Blob* vertexShaderBinary)
			{
				ID3D11VertexShader* pVertexShader;
				PGA_Rendering_D3D_checkedCall(device->CreateVertexShader(vertexShaderBinary->GetBufferPointer(), vertexShaderBinary->GetBufferSize(), 0, &pVertexShader));
				return pVertexShader;
			}

			ID3D11PixelShader* createPixelShader(ID3D11Device* device, ID3D10Blob* pixelShaderBinary)
			{
				ID3D11PixelShader* pPixelShader;
				PGA_Rendering_D3D_checkedCall(device->CreatePixelShader(pixelShaderBinary->GetBufferPointer(), pixelShaderBinary->GetBufferSize(), 0, &pPixelShader));
				return pPixelShader;
			}

			//////////////////////////////////////////////////////////////////////////
			void Shader::setIncludePath(const std::initializer_list<std::string>& includePaths)
			{
				g_includePaths.clear();
				for (auto& path : includePaths)
				{
					std::string newPath(path);
					if (path.empty())
						continue;
					// NOTE: guarantee that path always ends with a trailing slash (or backslash)
					if (path.back() != '/' && path.back() != '\\')
						g_includePaths.emplace_back(path + '\\');
					else
						g_includePaths.emplace_back(path);
				}
			}

			//////////////////////////////////////////////////////////////////////////
			Shader::Shader(ID3D11Device* device, ID3D10Blob* vertexShaderBinary, ID3D10Blob* pixelShaderBinary) :
				vertexShader(createVertexShader(device, vertexShaderBinary)),
				pixelShader(createPixelShader(device, pixelShaderBinary)),
				inputLayout(createInputLayout(device, vertexShaderBinary))
			{
			}

			Shader::Shader(ID3D11Device* device, const std::string& vertexShaderFileName, const std::string& pixelShaderFileName) :
				Shader(device, compileShader(vertexShaderFileName, "vs_5_0"), compileShader(pixelShaderFileName, "ps_5_0"))
			{

			}

			Shader::~Shader()
			{
				if (vertexShader)
					vertexShader->Release();
				if (pixelShader)
					pixelShader->Release();
				if (inputLayout)
					inputLayout->Release();
			}

			Shader::Shader(Shader&& other) :
				vertexShader(other.vertexShader),
				pixelShader(other.pixelShader),
				inputLayout(other.inputLayout)
			{
				vertexShader->AddRef();
				pixelShader->AddRef();
				inputLayout->AddRef();
			}

			Shader& Shader::operator=(Shader&& other)
			{
				vertexShader = other.vertexShader;
				pixelShader = other.pixelShader;
				inputLayout = other.inputLayout;
				vertexShader->AddRef();
				pixelShader->AddRef();
				inputLayout->AddRef();
				return *this;
			}

			ID3D11InputLayout* Shader::createInputLayout(ID3D11Device* device, ID3D10Blob* shaderBinary)
			{
				ID3D11InputLayout* pInputLayout;
				D3D11_INPUT_ELEMENT_DESC description[] = {
					{ "POSITION", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0 },
					{ "NORMAL", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, D3D11_APPEND_ALIGNED_ELEMENT, D3D11_INPUT_PER_VERTEX_DATA, 0 },
					{ "TEXCOORD", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, D3D11_APPEND_ALIGNED_ELEMENT, D3D11_INPUT_PER_VERTEX_DATA, 0 },
					{ "CUSTOM", 0, DXGI_FORMAT_R32_FLOAT, 0, D3D11_APPEND_ALIGNED_ELEMENT, D3D11_INPUT_PER_VERTEX_DATA, 0 },
					{ "MODELMATRIX", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 1, 0, D3D11_INPUT_PER_INSTANCE_DATA, 1 },
					{ "MODELMATRIX", 1, DXGI_FORMAT_R32G32B32A32_FLOAT, 1, D3D11_APPEND_ALIGNED_ELEMENT, D3D11_INPUT_PER_INSTANCE_DATA, 1 },
					{ "MODELMATRIX", 2, DXGI_FORMAT_R32G32B32A32_FLOAT, 1, D3D11_APPEND_ALIGNED_ELEMENT, D3D11_INPUT_PER_INSTANCE_DATA, 1 },
					{ "MODELMATRIX", 3, DXGI_FORMAT_R32G32B32A32_FLOAT, 1, D3D11_APPEND_ALIGNED_ELEMENT, D3D11_INPUT_PER_INSTANCE_DATA, 1 },
					{ "CUSTOM", 1, DXGI_FORMAT_R32_FLOAT, 1, D3D11_APPEND_ALIGNED_ELEMENT, D3D11_INPUT_PER_INSTANCE_DATA, 1 },

				};
				PGA_Rendering_D3D_checkedCall(device->CreateInputLayout(description, sizeof(description) / sizeof(description[0]), shaderBinary->GetBufferPointer(), shaderBinary->GetBufferSize(), &pInputLayout));
				return pInputLayout;
			}

			void Shader::bind(ID3D11DeviceContext* deviceContext)
			{
				deviceContext->VSSetShader(vertexShader, 0, 0);
				deviceContext->PSSetShader(pixelShader, 0, 0);
				deviceContext->IASetInputLayout(inputLayout);
			}

		}

	}

}