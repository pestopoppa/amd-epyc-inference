/**
 * @file render_math.cpp
 * @brief LaTeX to Unicode/ASCII math renderer
 *
 * Converts LaTeX mathematical expressions to Unicode or ASCII representation.
 *
 * Usage:
 *   {"command": "render_math", "latex": "\\frac{dy}{dx}", "format": "unicode"}
 *
 * Response:
 *   {"status": "success", "result": {"rendered": "dy/dx", "original": "\\frac{dy}{dx}"}}
 */

#include "../include/command.hpp"
#include <regex>
#include <map>
#include <sstream>

class RenderMathCommand : public Command {
public:
    std::string name() const override { return "render_math"; }
    std::string description() const override {
        return "Render LaTeX to Unicode/ASCII";
    }

    json execute(const json& params) override {
        Timer timer;

        try {
            if (!params.contains("latex")) {
                return error("Missing required parameter: latex");
            }

            std::string latex = params["latex"];
            std::string format = params.value("format", "unicode");
            bool use_unicode = (format == "unicode");

            std::string rendered = render(latex, use_unicode);

            json result;
            result["rendered"] = rendered;
            result["original"] = latex;
            result["format"] = format;

            json stats;
            stats["elapsed_ms"] = timer.elapsed_ms();

            return success(result, stats);

        } catch (const std::exception& e) {
            return error(std::string("Render error: ") + e.what());
        }
    }

private:
    // Greek letter mappings
    static const std::map<std::string, std::pair<std::string, std::string>>& greekLetters() {
        static const std::map<std::string, std::pair<std::string, std::string>> m = {
            // {latex, {unicode, ascii}}
            {"alpha", {"α", "alpha"}},
            {"beta", {"β", "beta"}},
            {"gamma", {"γ", "gamma"}},
            {"delta", {"δ", "delta"}},
            {"epsilon", {"ε", "epsilon"}},
            {"zeta", {"ζ", "zeta"}},
            {"eta", {"η", "eta"}},
            {"theta", {"θ", "theta"}},
            {"iota", {"ι", "iota"}},
            {"kappa", {"κ", "kappa"}},
            {"lambda", {"λ", "lambda"}},
            {"mu", {"μ", "mu"}},
            {"nu", {"ν", "nu"}},
            {"xi", {"ξ", "xi"}},
            {"pi", {"π", "pi"}},
            {"rho", {"ρ", "rho"}},
            {"sigma", {"σ", "sigma"}},
            {"tau", {"τ", "tau"}},
            {"upsilon", {"υ", "upsilon"}},
            {"phi", {"φ", "phi"}},
            {"chi", {"χ", "chi"}},
            {"psi", {"ψ", "psi"}},
            {"omega", {"ω", "omega"}},
            // Uppercase
            {"Alpha", {"Α", "Alpha"}},
            {"Beta", {"Β", "Beta"}},
            {"Gamma", {"Γ", "Gamma"}},
            {"Delta", {"Δ", "Delta"}},
            {"Theta", {"Θ", "Theta"}},
            {"Lambda", {"Λ", "Lambda"}},
            {"Xi", {"Ξ", "Xi"}},
            {"Pi", {"Π", "Pi"}},
            {"Sigma", {"Σ", "Sigma"}},
            {"Phi", {"Φ", "Phi"}},
            {"Psi", {"Ψ", "Psi"}},
            {"Omega", {"Ω", "Omega"}},
            // Variants
            {"varepsilon", {"ε", "epsilon"}},
            {"vartheta", {"ϑ", "theta"}},
            {"varphi", {"ϕ", "phi"}},
            {"varpi", {"ϖ", "pi"}},
        };
        return m;
    }

    // Symbol mappings
    static const std::map<std::string, std::pair<std::string, std::string>>& symbols() {
        static const std::map<std::string, std::pair<std::string, std::string>> m = {
            // Operators
            {"times", {"×", "*"}},
            {"div", {"÷", "/"}},
            {"cdot", {"·", "."}},
            {"pm", {"±", "+/-"}},
            {"mp", {"∓", "-/+"}},
            {"ast", {"∗", "*"}},
            {"star", {"⋆", "*"}},
            {"circ", {"∘", "o"}},
            {"bullet", {"•", "*"}},
            // Relations
            {"leq", {"≤", "<="}},
            {"geq", {"≥", ">="}},
            {"neq", {"≠", "!="}},
            {"approx", {"≈", "~="}},
            {"equiv", {"≡", "==="}},
            {"sim", {"∼", "~"}},
            {"propto", {"∝", "prop"}},
            {"ll", {"≪", "<<"}},
            {"gg", {"≫", ">>"}},
            // Arrows
            {"to", {"→", "->"}},
            {"rightarrow", {"→", "->"}},
            {"leftarrow", {"←", "<-"}},
            {"leftrightarrow", {"↔", "<->"}},
            {"Rightarrow", {"⇒", "=>"}},
            {"Leftarrow", {"⇐", "<="}},
            {"Leftrightarrow", {"⇔", "<=>"}},
            {"mapsto", {"↦", "|->"}},
            // Set theory
            {"in", {"∈", "in"}},
            {"notin", {"∉", "!in"}},
            {"subset", {"⊂", "subset"}},
            {"supset", {"⊃", "supset"}},
            {"subseteq", {"⊆", "subseteq"}},
            {"supseteq", {"⊇", "supseteq"}},
            {"cup", {"∪", "U"}},
            {"cap", {"∩", "^"}},
            {"emptyset", {"∅", "{}"}},
            // Logic
            {"forall", {"∀", "forall"}},
            {"exists", {"∃", "exists"}},
            {"neg", {"¬", "!"}},
            {"land", {"∧", "&&"}},
            {"lor", {"∨", "||"}},
            {"implies", {"⟹", "==>"}},
            {"iff", {"⟺", "<=>"}},
            // Calculus
            {"infty", {"∞", "inf"}},
            {"partial", {"∂", "d"}},
            {"nabla", {"∇", "nabla"}},
            {"int", {"∫", "integral"}},
            {"iint", {"∬", "double_integral"}},
            {"iiint", {"∭", "triple_integral"}},
            {"oint", {"∮", "contour_integral"}},
            {"sum", {"∑", "sum"}},
            {"prod", {"∏", "prod"}},
            // Misc
            {"sqrt", {"√", "sqrt"}},
            {"degree", {"°", "deg"}},
            {"prime", {"′", "'"}},
            {"dprime", {"″", "''"}},
            {"angle", {"∠", "angle"}},
            {"perp", {"⊥", "_|_"}},
            {"parallel", {"∥", "||"}},
            {"hbar", {"ℏ", "hbar"}},
            {"ell", {"ℓ", "l"}},
            {"Re", {"ℜ", "Re"}},
            {"Im", {"ℑ", "Im"}},
        };
        return m;
    }

    // Superscript mappings
    static const std::map<char, std::string>& superscripts() {
        static const std::map<char, std::string> m = {
            {'0', "⁰"}, {'1', "¹"}, {'2', "²"}, {'3', "³"}, {'4', "⁴"},
            {'5', "⁵"}, {'6', "⁶"}, {'7', "⁷"}, {'8', "⁸"}, {'9', "⁹"},
            {'+', "⁺"}, {'-', "⁻"}, {'=', "⁼"}, {'(', "⁽"}, {')', "⁾"},
            {'n', "ⁿ"}, {'i', "ⁱ"},
        };
        return m;
    }

    // Subscript mappings
    static const std::map<char, std::string>& subscripts() {
        static const std::map<char, std::string> m = {
            {'0', "₀"}, {'1', "₁"}, {'2', "₂"}, {'3', "₃"}, {'4', "₄"},
            {'5', "₅"}, {'6', "₆"}, {'7', "₇"}, {'8', "₈"}, {'9', "₉"},
            {'+', "₊"}, {'-', "₋"}, {'=', "₌"}, {'(', "₍"}, {')', "₎"},
            {'a', "ₐ"}, {'e', "ₑ"}, {'o', "ₒ"}, {'x', "ₓ"},
            {'i', "ᵢ"}, {'j', "ⱼ"}, {'k', "ₖ"}, {'n', "ₙ"}, {'m', "ₘ"},
        };
        return m;
    }

    std::string render(const std::string& latex, bool unicode) {
        std::string result = latex;

        // Remove \left and \right
        result = std::regex_replace(result, std::regex("\\\\left"), "");
        result = std::regex_replace(result, std::regex("\\\\right"), "");

        // Handle fractions: \frac{a}{b} -> a/b
        std::regex frac_re("\\\\frac\\{([^{}]*)\\}\\{([^{}]*)\\}");
        result = std::regex_replace(result, frac_re, "$1/$2");

        // Handle nested fractions (one more level)
        result = std::regex_replace(result, frac_re, "$1/$2");

        // Handle sqrt: \sqrt{x} -> √x or sqrt(x)
        std::regex sqrt_re("\\\\sqrt\\{([^{}]*)\\}");
        if (unicode) {
            result = std::regex_replace(result, sqrt_re, "√($1)");
        } else {
            result = std::regex_replace(result, sqrt_re, "sqrt($1)");
        }

        // Handle superscripts: x^{2} -> x²
        if (unicode) {
            std::regex sup_re("\\^\\{([^{}]*)\\}");
            std::smatch match;
            while (std::regex_search(result, match, sup_re)) {
                std::string content = match[1];
                std::string sup = toSuperscript(content);
                result = match.prefix().str() + sup + match.suffix().str();
            }

            // Single char superscripts: x^2 -> x²
            std::regex sup_single_re("\\^([0-9n])");
            while (std::regex_search(result, match, sup_single_re)) {
                char c = match[1].str()[0];
                auto it = superscripts().find(c);
                std::string sup = (it != superscripts().end()) ? it->second : match[1].str();
                result = match.prefix().str() + sup + match.suffix().str();
            }
        } else {
            // ASCII: keep ^{...}
            result = std::regex_replace(result, std::regex("\\^\\{([^{}]*)\\}"), "^($1)");
        }

        // Handle subscripts: x_{n} -> xₙ
        if (unicode) {
            std::regex sub_re("_\\{([^{}]*)\\}");
            std::smatch match;
            while (std::regex_search(result, match, sub_re)) {
                std::string content = match[1];
                std::string sub = toSubscript(content);
                result = match.prefix().str() + sub + match.suffix().str();
            }

            // Single char subscripts: x_0 -> x₀
            std::regex sub_single_re("_([0-9])");
            while (std::regex_search(result, match, sub_single_re)) {
                char c = match[1].str()[0];
                auto it = subscripts().find(c);
                std::string sub = (it != subscripts().end()) ? it->second : match[1].str();
                result = match.prefix().str() + sub + match.suffix().str();
            }
        } else {
            result = std::regex_replace(result, std::regex("_\\{([^{}]*)\\}"), "_($1)");
        }

        // Replace Greek letters
        for (const auto& [name, pair] : greekLetters()) {
            std::regex re("\\\\(" + name + ")(?![a-zA-Z])");
            result = std::regex_replace(result, re, unicode ? pair.first : pair.second);
        }

        // Replace symbols
        for (const auto& [name, pair] : symbols()) {
            std::regex re("\\\\(" + name + ")(?![a-zA-Z])");
            result = std::regex_replace(result, re, unicode ? pair.first : pair.second);
        }

        // Clean up remaining braces
        result = std::regex_replace(result, std::regex("\\{([^{}]*)\\}"), "$1");

        // Clean up spaces
        result = std::regex_replace(result, std::regex("\\\\,"), " ");
        result = std::regex_replace(result, std::regex("\\\\;"), " ");
        result = std::regex_replace(result, std::regex("\\\\:"), " ");
        result = std::regex_replace(result, std::regex("\\\\!"), "");
        result = std::regex_replace(result, std::regex("\\\\quad"), "  ");
        result = std::regex_replace(result, std::regex("\\\\qquad"), "    ");

        // Remove remaining backslashes before known text commands
        result = std::regex_replace(result, std::regex("\\\\text\\{([^{}]*)\\}"), "$1");
        result = std::regex_replace(result, std::regex("\\\\mathrm\\{([^{}]*)\\}"), "$1");
        result = std::regex_replace(result, std::regex("\\\\mathbf\\{([^{}]*)\\}"), "$1");

        return result;
    }

    std::string toSuperscript(const std::string& s) {
        std::string result;
        for (char c : s) {
            auto it = superscripts().find(c);
            if (it != superscripts().end()) {
                result += it->second;
            } else {
                result += c;  // Keep as-is if no mapping
            }
        }
        return result;
    }

    std::string toSubscript(const std::string& s) {
        std::string result;
        for (char c : s) {
            auto it = subscripts().find(c);
            if (it != subscripts().end()) {
                result += it->second;
            } else {
                result += c;  // Keep as-is if no mapping
            }
        }
        return result;
    }
};

// Factory function
std::unique_ptr<Command> create_render_math() {
    return std::make_unique<RenderMathCommand>();
}
