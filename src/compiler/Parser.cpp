#include "LineCounter.h"
#include "ParserAxis.h"
#include "ParserExpression.h"
#include "ParserOperand.h"
#include "ParserOperator.h"
#include "ParserParameterizedSuccessor.h"
#include "ParserProductionRule.h"
#include "ParserRand.h"
#include "ParserRepeatMode.h"
#include "ParserResult.h"
#include "ParserShapeAttribute.h"
#include "ParserSuccessor.h"
#include "ParserSymbol.h"
#include "ParserTerminal.h"
#include "Terminal.h"

#include <boost/algorithm/string.hpp>
#include <boost/config/warning_disable.hpp>
#include <boost/fusion/adapted/struct.hpp>
#include <boost/fusion/include/adapt_adt.hpp>
#include <boost/fusion/include/adapt_struct.hpp>
#include <boost/spirit/include/classic_position_iterator.hpp>
#include <boost/spirit/include/phoenix.hpp>
#include <boost/spirit/include/phoenix_core.hpp>
#include <boost/spirit/include/phoenix_fusion.hpp>
#include <boost/spirit/include/phoenix_object.hpp>
#include <boost/spirit/include/phoenix_operator.hpp>
#include <boost/spirit/include/phoenix_stl.hpp>
#include <boost/spirit/include/qi.hpp>
#include <boost/spirit/include/support_line_pos_iterator.hpp>
#include <boost/variant/recursive_variant.hpp>
#include <math/vector.h>
#include <pga/compiler/Parser.h>

#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>

#define BOOST_RESULT_OF_USE_DECLTYPE

namespace fusion = boost::fusion;
namespace phoenix = boost::phoenix;
namespace qi = boost::spirit::qi;
namespace ascii = boost::spirit::ascii;

namespace PGA
{
	namespace Compiler
	{
		namespace Parser
		{
			struct CommentSkipper : qi::grammar<std::string::const_iterator>
			{
				CommentSkipper() : CommentSkipper::base_type(skipRule, "C++-like comments")
				{
					using qi::lit;
					skipRule = ascii::space
						| (lit("//") >> *(qi::char_ - lit('\n')) >> -lit('\n'))
						| (lit("/*") >> *(qi::char_ - lit("*/")) > lit("*/"));
				}

				qi::rule<std::string::const_iterator> skipRule;

			};

			struct PGAGrammar : qi::grammar<std::string::const_iterator, PGA::Compiler::Parser::Result(), qi::locals<std::string>, CommentSkipper>
			{
				struct ErrorHandlingFunction
				{
					typedef qi::error_handler_result result_type;

					template <typename T1, typename T2, typename T3, typename T4>
					result_type operator()(T1 begin, T2 end, T3 where, T4 const& what) const
					{
						boost::spirit::line_pos_iterator<T3> line(begin);
						auto d = std::distance(begin, where);
						std::advance(line, d);
						auto currentLine = boost::spirit::get_current_line(begin, where, end);
						std::cerr << "Expecting " << what << " in line " << boost::spirit::get_line(line) <<
							" column: " << boost::spirit::get_column(std::begin(currentLine), where) << ": " << std::endl <<
							std::string(begin, std::begin(currentLine)) << std::endl <<
							"************************************" << std::endl <<
							std::string(std::begin(currentLine), std::end(currentLine)) << std::endl <<
							std::setw(std::distance(std::begin(currentLine), where)) << '^' << "---- here" << std::endl <<
							"************************************" << std::endl <<
							std::string(std::end(currentLine), end) << std::endl;
						return qi::fail;
					}

				};

				struct AnnotationFunction
				{
					typedef void result_type;

					template <typename, typename, typename>
					struct result
					{
						typedef void type;

					};

					AnnotationFunction(std::string::const_iterator codeBegin) : codeBegin(codeBegin) {}

					template <typename ValueT, typename FirstT, typename LastT>
					void operator()(ValueT& value, FirstT begin, LastT end) const 
					{
						boost::spirit::line_pos_iterator<std::string::const_iterator> line(codeBegin);
						auto d = std::distance(codeBegin, begin);
						std::advance(line, d);
						auto lineStart = boost::spirit::get_line_start(codeBegin, begin);
						value.line = boost::spirit::get_line(line);
						value.column = boost::spirit::get_column(lineStart, begin);
						value.length = std::distance(begin, end);
					}

				private:
					std::string::const_iterator codeBegin;

					static void annotate(...)
					{
						std::cerr << "(not having LocationInfo)\n";
					}

				};

				struct OperatorNamesList : qi::symbols<char, unsigned>
				{
					OperatorNamesList()
					{
						// NOTE: must be in the same order as OperatorType enumerator!
						add
							("Translate", TRANSLATE)
							("Rotate", ROTATE)
							("Scale", SCALE)
							("Extrude", EXTRUDE)
							("CompSplit", COMPSPLIT)
							("SubDiv", SUBDIV)
							("Repeat", REPEAT)
							("If", IF)
							("IfSizeLess", IFSIZELESS)
							("IfCollides", IFCOLLIDES)
							("RandomRule", STOCHASTIC)
							("SetAsDynamicConvexPolygon", SET_AS_DYNAMIC_CONVEX_POLYGON)
							("SetAsDynamicConvexRightPrism", SET_AS_DYNAMIC_CONVEX_RIGHT_PRISM)
							("SetAsDynamicConcavePolygon", SET_AS_DYNAMIC_CONCAVE_POLYGON)
							("SetAsDynamicConcaveRightPrism", SET_AS_DYNAMIC_CONCAVE_RIGHT_PRISM)
							("Collider", COLLIDER)
							("SwapSize", SWAPSIZE)
							("Replicate", REPLICATE)
							;
					}
				} operatorNames;

				struct NoSuccessorOperatorNamesList : qi::symbols < char, unsigned >
				{
					NoSuccessorOperatorNamesList()
					{
						add
							("Discard", DISCARD)
							("Generate", GENERATE)
							;
					}
				} terminalOperatorNames;

				struct ShapeTypeNamesList : qi::symbols<char, unsigned>
				{
					ShapeTypeNamesList()
					{
						// NOTE: must be in the same order as ShapeType enumerator!
						add
							("Triangle", TRIANGLE)
							("Quad", QUAD)
							("Pentagon", PENTAGON)
							("Hexagon", HEXAGON)
							("Heptagon", HEPTAGON)
							("Octagon", OCTAGON)
							("Prism3", PRISM3)
							("Box", BOX)
							("Prism5", PRISM5)
							("Prism6", PRISM6)
							("Prism7", PRISM7)
							("Prism8", PRISM8)
							("DynamicConvexRightPrism", DYNAMIC_CONVEX_RIGHT_PRISM)
							("DynamicRightPrism", DYNAMIC_RIGHT_PRISM)
							("DynamicConvexPolygon", DYNAMIC_CONVEX_POLYGON)
							("DynamicPolygon", DYNAMIC_POLYGON)
							("Sphere", SPHERE)
							;
					}
				} shapeTypeNames;

				struct ShapeAttributeNamesList : qi::symbols<char, unsigned>
				{
					ShapeAttributeNamesList()
					{
						// NOTE: must be in the same order as ShapeAttribute enumerator!
						add
							("ShapePos", SHAPE_POS)
							("ShapeSize", SHAPE_SIZE)
							("ShapeRotation", SHAPE_ROTATION)
							("ShapeNormal", SHAPE_NORMAL)
							("ShapeSeed", SHAPE_SEED)
							("ShapeCustomAttribute", SHAPE_CUSTOM_ATTR)
							;
					}
				} shapeAttributeNames;

				struct AxisNamesList : qi::symbols<char, float>
				{
					AxisNamesList()
					{
						// NOTE: must be in the same order as PGA::Axis enumerator!
						add
							("X", 0)
							("Y", 1)
							("Z", 2)
							;
					}
				} axisNames;

				struct RepeatModeNamesList : qi::symbols < char, float >
				{
					RepeatModeNamesList()
					{
						// NOTE: must be in the same order as PGA::RepeatMode enumerator!
						add
							("ANCHOR_TO_START", 0)
							("ANCHOR_TO_END", 1)
							("ADJUST_TO_FILL", 2)
							;
					}
				} repeatModeNames;

				struct OperationsNamesList : qi::symbols <char, unsigned>
				{
					// NOTE: must be in the same order as PGA::OperationType enumerator!
					OperationsNamesList()
					{
						add
							("+", PGA::OperationType::OPT_ADD)
							("-", PGA::OperationType::OPT_SUB)
							("*", PGA::OperationType::OPT_MULTI)
							("/", PGA::OperationType::OPT_DIV)
							("==", PGA::OperationType::OPT_EQ)
							("!=", PGA::OperationType::OPT_NEQ)
							("<", PGA::OperationType::OPT_LT)
							(">", PGA::OperationType::OPT_GT)
							("<=", PGA::OperationType::OPT_LEQ)
							(">=", PGA::OperationType::OPT_GEQ)
							("&&", PGA::OperationType::OPT_AND)
							("||", PGA::OperationType::OPT_OR)
							;
					}
				} operationsNamesList;

				phoenix::function<ErrorHandlingFunction> errorHandler;
				phoenix::function<AnnotationFunction> annotator;
				qi::rule<std::string::const_iterator, math::float2(), CommentSkipper> axiomVertex;
				qi::rule<std::string::const_iterator, double(), CommentSkipper> doubleOrVariable;
				qi::rule<std::string::const_iterator, PGA::Compiler::Parser::Axiom(), CommentSkipper> axiom;
				qi::rule<std::string::const_iterator, PGA::Compiler::Parser::Terminal(), CommentSkipper> terminal;
				qi::rule<std::string::const_iterator, PGA::Compiler::Parser::Symbol(), CommentSkipper> symbol;
				qi::rule<std::string::const_iterator, std::string(), CommentSkipper> identifier;
				qi::rule<std::string::const_iterator, void(), CommentSkipper> productionSymbol;
				qi::rule<std::string::const_iterator, double(), CommentSkipper> probability;
				qi::rule<std::string::const_iterator, PGA::Compiler::Parser::Axis(), CommentSkipper> axis;
				qi::rule<std::string::const_iterator, PGA::Compiler::Parser::RepeatMode(), CommentSkipper> repeatMode;
				qi::rule<std::string::const_iterator, PGA::Compiler::Parser::ShapeAttribute(), CommentSkipper> shapeAttribute;
				qi::rule<std::string::const_iterator, PGA::Compiler::Parser::Rand(), CommentSkipper> rand;
				qi::rule<std::string::const_iterator, PGA::Compiler::Parser::Operand(), CommentSkipper> operand;
				qi::rule<std::string::const_iterator, PGA::Compiler::Parser::Expression(), CommentSkipper> expression;
				qi::rule<std::string::const_iterator, PGA::Compiler::Parser::Vec2(), CommentSkipper> vec2;
				qi::rule<std::string::const_iterator, PGA::Compiler::Parser::Operator(), CommentSkipper> operatorWithSuccessor;
				qi::rule<std::string::const_iterator, PGA::Compiler::Parser::Operator(), CommentSkipper> operatorWithoutSuccessor;
				qi::rule<std::string::const_iterator, PGA::Compiler::Parser::Successor(), CommentSkipper> successor;
				qi::rule<std::string::const_iterator, PGA::Compiler::Parser::ParameterizedSuccessor(), CommentSkipper> parameterizedSuccessor;
				qi::rule<std::string::const_iterator, PGA::Compiler::Parser::ProductionRule(), CommentSkipper> productionRule;
				qi::rule<std::string::const_iterator, void(), CommentSkipper> variable;
				qi::rule<std::string::const_iterator, PGA::Compiler::Parser::Result(), qi::locals<std::string>, CommentSkipper> grammar;
				std::map<std::string, double> variables;

				PGAGrammar(std::string::const_iterator& codeStart) : PGAGrammar::base_type(grammar, "PGA Grammar"), annotator(codeStart)
				{
					using qi::lit;
					using qi::lexeme;
					using qi::on_error;
					using qi::fail;
					using ascii::char_;
					using ascii::alnum;
					using ascii::alpha;
					using qi::_1;
					using qi::_2;
					using qi::_3;
					using qi::_4;
					using ascii::string;
					using boost::spirit::ascii::space;
					using namespace qi::labels;
					using phoenix::construct;
					using phoenix::val;
					using phoenix::at_c;
					using phoenix::push_back;

					doubleOrVariable =
					(
						qi::double_[_val = _1] | identifier[_val = phoenix::ref(variables)[_1]]
					);

					identifier = 
					(
						lexeme[+qi::char_("a-zA-Z_0-9_")[_val += _1]]
					);

					variable =
						(
						lit("var") >
						(identifier >
						lit("=") >
						doubleOrVariable)[phoenix::ref(variables)[_1] = _2] >
						lit(";")
						);

					axiomVertex = 
					(
						(doubleOrVariable >> ',' >> doubleOrVariable)[_val = construct<math::float2>(_1, _2)]
					);

					axiom = 
					(
						lit("axiom") >
						shapeTypeNames[at_c<1>(_val) = _1] >
						*('{' > *(lit('(') > axiomVertex[push_back(at_c<2>(_val), _1)] > lit(')') > *lit(',')) > '}') >
						identifier[at_c<0>(_val) = _1] > 
						lit(";")
					);

					terminal = 
					(
						lit("terminal") > 
						+identifier[push_back(at_c<0>(_val), _1)] > lit("(") >> qi::double_[push_back(at_c<1>(_val), _1)] >>
						*(lit(",") > qi::double_[push_back(at_c<1>(_val), _1)]) > lit(")") >> 
						lit(";")
					);

					probability = 
					(
						qi::double_[_val = _1] >> 
						lit("%") >> 
						lit(":") | 
						(
							lit("else") >> 
							lit(":")
						)
						[_val = -1]
					);

					axis = axisNames[at_c<0>(_val) = _1];

					repeatMode = repeatModeNames[at_c<0>(_val) = _1];

					shapeAttribute = 
					(
						shapeAttributeNames[at_c<0>(_val) = _1] > lit("(") > -(axisNames[at_c<1>(_val) = _1] > *(lit(",") > axisNames[at_c<2>(_val) = _1])) > lit(")")
					);

					rand = 
					(
						lit("Rand(") > doubleOrVariable[at_c<0>(_val) = _1] > lit(",") > doubleOrVariable[at_c<1>(_val) = _1] > lit(")")
					);

					vec2 = 
					(
						lit("(") > doubleOrVariable[at_c<0>(_val) = _1] > lit(",") > doubleOrVariable[at_c<1>(_val) = _1] > lit(")")
					);

					operand = 
					(
						axis |
						repeatMode |
						rand |
						shapeAttribute |
						vec2 |
						expression |
						doubleOrVariable
					);

					expression = 
					(
						lit("Exp(") > 
							operand[at_c<1>(_val) = _1] >
							operationsNamesList[at_c<0>(_val) = _1] > 
							operand[at_c<2>(_val) = _1] >
						lit(")")
					);

					symbol = 
					(
						(probability[at_c<1>(_val) = _1] > identifier[at_c<0>(_val) = _1]) | 
						identifier[at_c<0>(_val) = _1, at_c<1>(_val) = 0]
					);

					successor = 
					(
						operatorWithSuccessor | 
						operatorWithoutSuccessor | 
						symbol
					);

					parameterizedSuccessor = 
					(
						doubleOrVariable[push_back(at_c<0>(_val), _1)] >> lit(":") >> successor[at_c<1>(_val) = _1]
					);

					operatorWithSuccessor = 
					(
						-probability[at_c<3>(_val) = _1] >>
						operatorNames[at_c<0>(_val) = _1] >>
						lit("(") >>
								-operand[push_back(at_c<1>(_val), _1)] ||
								*(lit(",") >> operand[push_back(at_c<1>(_val), _1)]) >> 
						lit(")") >>
						(
							(
								lit("{") >>
									parameterizedSuccessor[push_back(at_c<2>(_val), _1)] >>
									*(lit("|") >> parameterizedSuccessor[push_back(at_c<2>(_val), _1)]) >>
								lit("}")
							) |
							(
								lit("{") >>
									successor[push_back(at_c<2>(_val), _1)] >>
									*(lit("|") >> successor[push_back(at_c<2>(_val), _1)]) >>
								lit("}")
							) | 
							successor[push_back(at_c<2>(_val), _1)]
						)
					);

					operatorWithoutSuccessor = 
					(
						terminalOperatorNames[at_c<0>(_val) = _1] >> lit('(') >> lit(')')
					);

					productionSymbol = 
					(
						lit("-->") | 
						lit("->") | 
						lit("==>") | 
						lit("=>") | 
						lit("=") | 
						lit(">>")
					);

					productionRule = 
					(
						identifier[at_c<0>(_val) = _1] > 
						productionSymbol > 
						+successor[push_back(at_c<1>(_val), _1)] > 
						lit(";")
					);

					grammar =
						(*variable) ||
						(+axiom[push_back(at_c<0>(_val), _1)]) ||
						(*terminal[push_back(at_c<2>(_val), _1)]) ||
						(+productionRule[push_back(at_c<1>(_val), _1)]);

					axiomVertex.name("[axiom vertex]");
					axiom.name("[axiom]");
					terminal.name("[terminal]");
					identifier.name("[symbol name]");
					symbol.name("[symbol]");
					successor.name("[successor]");
					parameterizedSuccessor.name("[parameterized successor]");
					operatorWithoutSuccessor.name("[no successor production]");
					operatorWithSuccessor.name("[operator]");
					productionRule.name("[production rule]");
					productionSymbol.name("[production symbol]");
					probability.name("[probability]");
					rand.name("[rand]");
					expression.name("[expression]");
					operand.name("[operand]");
					variable.name("[variable]");
					grammar.name("[grammar]");

					auto setLocationInfo = annotator(_val, _1, _3);

					on_success(operatorWithSuccessor, setLocationInfo);
					on_success(parameterizedSuccessor, setLocationInfo);
					on_success(productionRule, setLocationInfo);

					on_error<fail>(axiom, errorHandler(_1, _2, _3, _4));
					on_error<fail>(terminal, errorHandler(_1, _2, _3, _4));
					on_error<fail>(operatorWithSuccessor, errorHandler(_1, _2, _3, _4));
					on_error<fail>(probability, errorHandler(_1, _2, _3, _4));
					on_error<fail>(parameterizedSuccessor, errorHandler(_1, _2, _3, _4));
					on_error<fail>(rand, errorHandler(_1, _2, _3, _4));
					on_error<fail>(grammar, errorHandler(_1, _2, _3, _4));
					on_error<fail>(symbol, errorHandler(_1, _2, _3, _4));
				}

			};

			bool parse(const std::string& sourceCode, Logger& logger, std::vector<PGA::Compiler::Axiom>& axioms, std::vector<PGA::Compiler::Rule>& rules)
			{
				std::string::const_iterator codeBegin = sourceCode.begin();
				std::string::const_iterator codeEnd = sourceCode.end();

				Result result;
				PGAGrammar grammar(codeBegin);
				CommentSkipper skipper;

				bool parsed = false;
				try
				{
					parsed = phrase_parse(codeBegin, codeEnd, grammar, skipper, result);
				}
				catch (const qi::expectation_failure<std::string::const_iterator>& e)
				{
					size_t line, column;
					LineCounter::count(codeBegin, e.first, line, column);
					std::stringstream stream;
					stream << e.what_;
					logger.addMessage(Logger::LL_ERROR, "(line = %d, column = %d) expecting %s", line, column, stream.str().c_str());
					return false;
				}

				if (parsed && codeBegin == codeEnd)
				{
					if (!result.check(logger))
						return false;
					// TODO:
					std::vector<PGA::Compiler::Terminal> terminals;
					result.convertToAbstraction(axioms, rules, terminals);
					return true;
				}
				else
					return false;
			}

		}

	}

}

BOOST_FUSION_ADAPT_STRUCT(
	PGA::Compiler::Parser::Expression,
	(unsigned int, operation)
	(PGA::Compiler::Parser::Operand, left)
	(PGA::Compiler::Parser::Operand, right)
)

BOOST_FUSION_ADAPT_STRUCT(
	PGA::Compiler::Parser::Axis,
	(unsigned int, type)
)

BOOST_FUSION_ADAPT_STRUCT(
	PGA::Compiler::Parser::RepeatMode,
	(unsigned int, type)
)

BOOST_FUSION_ADAPT_STRUCT(
	PGA::Compiler::Parser::Vec2,
	(double, x)
	(double, y)
)

BOOST_FUSION_ADAPT_STRUCT(
	PGA::Compiler::Parser::ShapeAttribute,
	(unsigned int, type)
	(double, axis)
	(double, component)
)

BOOST_FUSION_ADAPT_STRUCT(
	PGA::Compiler::Parser::Rand,
	(double, min)
	(double, max)
)

BOOST_FUSION_ADAPT_STRUCT(
	PGA::Compiler::Parser::Symbol,
	(std::string, symbol)
	(double, probability)
)

BOOST_FUSION_ADAPT_STRUCT(
	PGA::Compiler::Parser::Axiom,
	(std::string, name)
	(unsigned int, shapeType)
	(std::vector<::math::float2>, vertices)
)

BOOST_FUSION_ADAPT_STRUCT(
	PGA::Compiler::Parser::Terminal,
	(std::vector<std::string>, symbols)
	(std::vector<double>, parameters)
)

BOOST_FUSION_ADAPT_STRUCT(
	PGA::Compiler::Parser::Operator,
	(unsigned int, operator_)
	(std::vector<PGA::Compiler::Parser::Operand>, parameters)
	(std::list<PGA::Compiler::Parser::ParameterizedSuccessor>, successors)
	(double, probability)
)

BOOST_FUSION_ADAPT_STRUCT(
	PGA::Compiler::Parser::ProductionRule,
	(std::string, symbol)
	(std::vector<PGA::Compiler::Parser::Successor>, successors)
)

BOOST_FUSION_ADAPT_STRUCT(
	PGA::Compiler::Parser::ParameterizedSuccessor,
	(std::vector<PGA::Compiler::Parser::Operand>, parameters)
	(PGA::Compiler::Parser::Successor, successor)
)

BOOST_FUSION_ADAPT_STRUCT(
	PGA::Compiler::Parser::Result,
	(std::list<PGA::Compiler::Parser::Axiom>, axioms)
	(std::list<PGA::Compiler::Parser::ProductionRule>, rules)
	(std::list<PGA::Compiler::Parser::Terminal>, terminals)
)
