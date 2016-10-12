#pragma once

//////////////////////////////////////////////////////////////////////////
int main(unsigned int argc, const char** argv);

//////////////////////////////////////////////////////////////////////////
class OperatorGraphAnalyzerApp
{
public:
    int run(unsigned int argc, const char** argv);

    friend int main(unsigned int argc, const char** argv);
    
private:
    OperatorGraphAnalyzerApp() = default;
    ~OperatorGraphAnalyzerApp() = default;
    
};
