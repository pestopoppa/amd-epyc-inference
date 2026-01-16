/**
 * @file expression.hpp
 * @brief Simple expression parser for mathematical expressions
 *
 * Supports: +, -, *, /, **, sin, cos, exp, log, sqrt, tanh
 * Variables: x0, x1, x2, ... (indexed)
 */

#pragma once

#include <string>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <sstream>
#include <cctype>
#include <map>

class Expression {
public:
    explicit Expression(const std::string& expr) : expr_(expr), pos_(0) {}

    /**
     * @brief Evaluate expression with given variable values
     * @param vars Vector of variable values [x0, x1, x2, ...]
     * @return Evaluated result
     */
    double evaluate(const std::vector<double>& vars) {
        vars_ = &vars;
        pos_ = 0;
        return parseExpression();
    }

private:
    std::string expr_;
    size_t pos_;
    const std::vector<double>* vars_;

    char peek() const {
        skipSpaces();
        return pos_ < expr_.size() ? expr_[pos_] : '\0';
    }

    char get() {
        skipSpaces();
        return pos_ < expr_.size() ? expr_[pos_++] : '\0';
    }

    void skipSpaces() const {
        while (pos_ < expr_.size() && std::isspace(expr_[pos_])) {
            const_cast<size_t&>(pos_)++;
        }
    }

    double parseExpression() {
        double result = parseTerm();
        while (true) {
            char op = peek();
            if (op == '+') {
                get();
                result += parseTerm();
            } else if (op == '-') {
                get();
                result -= parseTerm();
            } else {
                break;
            }
        }
        return result;
    }

    double parseTerm() {
        double result = parsePower();
        while (true) {
            char op = peek();
            if (op == '*' && pos_ + 1 < expr_.size() && expr_[pos_ + 1] != '*') {
                get();
                result *= parsePower();
            } else if (op == '/') {
                get();
                result /= parsePower();
            } else {
                break;
            }
        }
        return result;
    }

    double parsePower() {
        double base = parseUnary();
        if (pos_ + 1 < expr_.size() && expr_[pos_] == '*' && expr_[pos_ + 1] == '*') {
            pos_ += 2;
            double exp = parsePower();  // Right-associative
            return std::pow(base, exp);
        }
        return base;
    }

    double parseUnary() {
        char op = peek();
        if (op == '-') {
            get();
            return -parseUnary();
        } else if (op == '+') {
            get();
            return parseUnary();
        }
        return parsePrimary();
    }

    double parsePrimary() {
        char c = peek();

        // Parentheses
        if (c == '(') {
            get();
            double result = parseExpression();
            if (get() != ')') {
                throw std::runtime_error("Expected ')'");
            }
            return result;
        }

        // Number
        if (std::isdigit(c) || c == '.') {
            return parseNumber();
        }

        // Variable or function
        if (std::isalpha(c) || c == '_') {
            return parseIdentifier();
        }

        throw std::runtime_error("Unexpected character: " + std::string(1, c));
    }

    double parseNumber() {
        size_t start = pos_;
        while (pos_ < expr_.size() && (std::isdigit(expr_[pos_]) || expr_[pos_] == '.' || expr_[pos_] == 'e' || expr_[pos_] == 'E' ||
               ((expr_[pos_] == '+' || expr_[pos_] == '-') && pos_ > 0 && (expr_[pos_-1] == 'e' || expr_[pos_-1] == 'E')))) {
            pos_++;
        }
        return std::stod(expr_.substr(start, pos_ - start));
    }

    double parseIdentifier() {
        size_t start = pos_;
        while (pos_ < expr_.size() && (std::isalnum(expr_[pos_]) || expr_[pos_] == '_')) {
            pos_++;
        }
        std::string name = expr_.substr(start, pos_ - start);

        // Check for variable (x0, x1, x2, ...)
        if (name[0] == 'x' && name.size() > 1) {
            try {
                size_t idx = std::stoul(name.substr(1));
                if (idx < vars_->size()) {
                    return (*vars_)[idx];
                }
                throw std::runtime_error("Variable index out of range: " + name);
            } catch (...) {
                // Not a variable, continue to check functions
            }
        }

        // Single variable 'x' or 'y' shorthand
        if (name == "x" && !vars_->empty()) return (*vars_)[0];
        if (name == "y" && vars_->size() > 1) return (*vars_)[1];

        // Constants
        if (name == "pi" || name == "PI") return M_PI;
        if (name == "e" || name == "E") return M_E;

        // Functions (require parentheses)
        if (peek() == '(') {
            get();  // consume '('
            double arg = parseExpression();
            if (get() != ')') {
                throw std::runtime_error("Expected ')' after function argument");
            }

            if (name == "sin") return std::sin(arg);
            if (name == "cos") return std::cos(arg);
            if (name == "tan") return std::tan(arg);
            if (name == "exp") return std::exp(arg);
            if (name == "log" || name == "ln") return std::log(arg);
            if (name == "log10") return std::log10(arg);
            if (name == "sqrt") return std::sqrt(arg);
            if (name == "abs") return std::abs(arg);
            if (name == "tanh") return std::tanh(arg);
            if (name == "sinh") return std::sinh(arg);
            if (name == "cosh") return std::cosh(arg);
            if (name == "floor") return std::floor(arg);
            if (name == "ceil") return std::ceil(arg);

            throw std::runtime_error("Unknown function: " + name);
        }

        throw std::runtime_error("Unknown identifier: " + name);
    }
};
