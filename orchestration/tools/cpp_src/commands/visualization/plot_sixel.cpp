/**
 * @file plot_sixel.cpp
 * @brief High-resolution Sixel graphics plotting
 *
 * Creates plots using the Sixel graphics protocol for terminals that support it
 * (xterm, mlterm, WezTerm, etc.). Falls back to braille for unsupported terminals.
 *
 * Usage:
 *   {"command": "plot_sixel", "x": [0,1,2,3,4], "y": [0,1,4,9,16], "title": "y = x²"}
 *
 * Response:
 *   {"status": "success", "result": {"sixel": "...", "format": "sixel", ...}}
 */

#include "../include/command.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <sstream>
#include <iomanip>

class PlotSixelCommand : public Command {
public:
    std::string name() const override { return "plot_sixel"; }
    std::string description() const override {
        return "Create high-resolution sixel graphics plot";
    }

    json execute(const json& params) override {
        Timer timer;

        try {
            // Parse parameters
            if (!params.contains("x") || !params.contains("y")) {
                return error("Missing required parameters: x and y");
            }

            std::vector<double> x = params["x"].get<std::vector<double>>();
            std::vector<double> y = params["y"].get<std::vector<double>>();

            if (x.size() != y.size()) {
                return error("x and y must have the same length");
            }
            if (x.empty()) {
                return error("x and y cannot be empty");
            }

            std::string plot_type = params.value("type", "line");
            int width = params.value("width", 800);
            int height = params.value("height", 400);
            std::string title = params.value("title", "");
            std::string x_label = params.value("x_label", "");
            std::string y_label = params.value("y_label", "");

            // Clamp dimensions
            width = std::clamp(width, 100, 2000);
            height = std::clamp(height, 50, 1000);

            // Check terminal support
            bool sixel_supported = checkSixelSupport();

            std::string output;
            std::string format;

            if (sixel_supported) {
                output = generateSixel(x, y, plot_type, width, height, title);
                format = "sixel";
            } else {
                // Fallback to braille
                output = generateBraille(x, y, plot_type, width / 10, height / 5, title);
                format = "braille";
            }

            // Calculate data ranges
            double x_min = *std::min_element(x.begin(), x.end());
            double x_max = *std::max_element(x.begin(), x.end());
            double y_min = *std::min_element(y.begin(), y.end());
            double y_max = *std::max_element(y.begin(), y.end());

            json result;
            if (format == "sixel") {
                result["sixel"] = output;
            } else {
                result["plot"] = output;
            }
            result["format"] = format;
            result["dimensions"]["width"] = width;
            result["dimensions"]["height"] = height;
            result["x_range"] = {x_min, x_max};
            result["y_range"] = {y_min, y_max};
            result["n_points"] = static_cast<int>(x.size());

            json stats;
            stats["elapsed_ms"] = timer.elapsed_ms();
            stats["terminal_sixel_support"] = sixel_supported;

            return success(result, stats);

        } catch (const std::exception& e) {
            return error(std::string("Plot error: ") + e.what());
        }
    }

private:
    /**
     * @brief Check if terminal supports Sixel graphics
     */
    bool checkSixelSupport() {
        // Check TERM environment variable
        const char* term = std::getenv("TERM");
        if (!term) return false;

        std::string term_str(term);

        // Known Sixel-supporting terminals
        if (term_str.find("xterm") != std::string::npos ||
            term_str.find("mlterm") != std::string::npos ||
            term_str.find("mintty") != std::string::npos ||
            term_str.find("wezterm") != std::string::npos ||
            term_str.find("foot") != std::string::npos ||
            term_str.find("contour") != std::string::npos) {
            return true;
        }

        // Check COLORTERM for additional hints
        const char* colorterm = std::getenv("COLORTERM");
        if (colorterm && std::string(colorterm) == "truecolor") {
            // Modern terminal, might support sixel
            // Could do DA1 query here, but for now be conservative
        }

        return false;
    }

    /**
     * @brief Generate Sixel graphics output
     */
    std::string generateSixel(const std::vector<double>& x,
                               const std::vector<double>& y,
                               const std::string& plot_type,
                               int width, int height,
                               const std::string& title) {
        // Create pixel buffer (RGB)
        std::vector<std::vector<std::array<uint8_t, 3>>> pixels(
            height, std::vector<std::array<uint8_t, 3>>(width, {255, 255, 255}));

        // Calculate scaling
        double x_min = *std::min_element(x.begin(), x.end());
        double x_max = *std::max_element(x.begin(), x.end());
        double y_min = *std::min_element(y.begin(), y.end());
        double y_max = *std::max_element(y.begin(), y.end());

        // Add margins
        int margin_left = 50;
        int margin_right = 20;
        int margin_top = title.empty() ? 10 : 30;
        int margin_bottom = 30;

        int plot_width = width - margin_left - margin_right;
        int plot_height = height - margin_top - margin_bottom;

        // Avoid division by zero
        double x_scale = (x_max > x_min) ? plot_width / (x_max - x_min) : 1.0;
        double y_scale = (y_max > y_min) ? plot_height / (y_max - y_min) : 1.0;

        // Draw axes (gray)
        std::array<uint8_t, 3> axis_color = {128, 128, 128};
        // X-axis
        int y_axis = height - margin_bottom;
        for (int px = margin_left; px < width - margin_right; ++px) {
            pixels[y_axis][px] = axis_color;
        }
        // Y-axis
        for (int py = margin_top; py < height - margin_bottom; ++py) {
            pixels[py][margin_left] = axis_color;
        }

        // Draw data
        std::array<uint8_t, 3> data_color = {0, 100, 200};  // Blue

        if (plot_type == "scatter") {
            // Scatter plot
            for (size_t i = 0; i < x.size(); ++i) {
                int px = margin_left + static_cast<int>((x[i] - x_min) * x_scale);
                int py = height - margin_bottom - static_cast<int>((y[i] - y_min) * y_scale);
                px = std::clamp(px, margin_left, width - margin_right - 1);
                py = std::clamp(py, margin_top, height - margin_bottom - 1);

                // Draw a small circle
                for (int dy = -2; dy <= 2; ++dy) {
                    for (int dx = -2; dx <= 2; ++dx) {
                        if (dx*dx + dy*dy <= 4) {
                            int pxx = px + dx;
                            int pyy = py + dy;
                            if (pxx >= 0 && pxx < width && pyy >= 0 && pyy < height) {
                                pixels[pyy][pxx] = data_color;
                            }
                        }
                    }
                }
            }
        } else {
            // Line plot
            for (size_t i = 0; i + 1 < x.size(); ++i) {
                int x1 = margin_left + static_cast<int>((x[i] - x_min) * x_scale);
                int y1 = height - margin_bottom - static_cast<int>((y[i] - y_min) * y_scale);
                int x2 = margin_left + static_cast<int>((x[i+1] - x_min) * x_scale);
                int y2 = height - margin_bottom - static_cast<int>((y[i+1] - y_min) * y_scale);

                drawLine(pixels, x1, y1, x2, y2, data_color);
            }
        }

        // Convert to Sixel format
        return pixelsToSixel(pixels);
    }

    /**
     * @brief Draw a line using Bresenham's algorithm
     */
    void drawLine(std::vector<std::vector<std::array<uint8_t, 3>>>& pixels,
                  int x1, int y1, int x2, int y2,
                  const std::array<uint8_t, 3>& color) {
        int height = pixels.size();
        int width = pixels[0].size();

        int dx = std::abs(x2 - x1);
        int dy = std::abs(y2 - y1);
        int sx = (x1 < x2) ? 1 : -1;
        int sy = (y1 < y2) ? 1 : -1;
        int err = dx - dy;

        while (true) {
            if (x1 >= 0 && x1 < width && y1 >= 0 && y1 < height) {
                pixels[y1][x1] = color;
                // Draw thicker line
                if (y1 + 1 < height) pixels[y1 + 1][x1] = color;
                if (x1 + 1 < width) pixels[y1][x1 + 1] = color;
            }

            if (x1 == x2 && y1 == y2) break;

            int e2 = 2 * err;
            if (e2 > -dy) {
                err -= dy;
                x1 += sx;
            }
            if (e2 < dx) {
                err += dx;
                y1 += sy;
            }
        }
    }

    /**
     * @brief Convert pixel buffer to Sixel format
     */
    std::string pixelsToSixel(const std::vector<std::vector<std::array<uint8_t, 3>>>& pixels) {
        int height = pixels.size();
        int width = pixels[0].size();

        std::ostringstream ss;

        // Sixel header
        ss << "\033Pq";  // DCS (Device Control String) for Sixel

        // Define color palette (simplified: just a few colors)
        // Color 0: white (background)
        ss << "#0;2;100;100;100";
        // Color 1: blue (data)
        ss << "#1;2;0;40;80";
        // Color 2: gray (axes)
        ss << "#2;2;50;50;50";
        // Color 3: black (text)
        ss << "#3;2;0;0;0";

        // Convert pixels to sixel rows (each sixel is 6 pixels tall)
        for (int row = 0; row < height; row += 6) {
            // For each color
            for (int color_id = 1; color_id <= 2; ++color_id) {
                ss << "#" << color_id;

                for (int col = 0; col < width; ++col) {
                    int sixel_val = 0;

                    // Check 6 vertical pixels
                    for (int bit = 0; bit < 6 && row + bit < height; ++bit) {
                        const auto& pixel = pixels[row + bit][col];

                        bool is_this_color = false;
                        if (color_id == 1 && pixel[2] > 150) {
                            // Blue-ish (data)
                            is_this_color = true;
                        } else if (color_id == 2 && pixel[0] == 128 && pixel[1] == 128) {
                            // Gray (axes)
                            is_this_color = true;
                        }

                        if (is_this_color) {
                            sixel_val |= (1 << bit);
                        }
                    }

                    // Sixel character is value + 63
                    ss << static_cast<char>(sixel_val + 63);
                }

                ss << "$";  // Carriage return (go back to start of line)
            }

            ss << "-";  // New line (move down 6 pixels)
        }

        // Sixel terminator
        ss << "\033\\";

        return ss.str();
    }

    /**
     * @brief Generate braille fallback plot
     */
    std::string generateBraille(const std::vector<double>& x,
                                 const std::vector<double>& y,
                                 const std::string& plot_type,
                                 int width, int height,
                                 const std::string& title) {
        // Braille characters: each char is 2x4 dots
        // Unicode braille starts at U+2800

        // Create dot grid (2x width, 4x height per character)
        int dot_width = width * 2;
        int dot_height = height * 4;
        std::vector<std::vector<bool>> dots(dot_height, std::vector<bool>(dot_width, false));

        // Calculate scaling
        double x_min = *std::min_element(x.begin(), x.end());
        double x_max = *std::max_element(x.begin(), x.end());
        double y_min = *std::min_element(y.begin(), y.end());
        double y_max = *std::max_element(y.begin(), y.end());

        double x_scale = (x_max > x_min) ? (dot_width - 1) / (x_max - x_min) : 1.0;
        double y_scale = (y_max > y_min) ? (dot_height - 1) / (y_max - y_min) : 1.0;

        // Plot points
        for (size_t i = 0; i < x.size(); ++i) {
            int px = static_cast<int>((x[i] - x_min) * x_scale);
            int py = dot_height - 1 - static_cast<int>((y[i] - y_min) * y_scale);
            px = std::clamp(px, 0, dot_width - 1);
            py = std::clamp(py, 0, dot_height - 1);
            dots[py][px] = true;
        }

        // If line plot, connect dots
        if (plot_type == "line") {
            for (size_t i = 0; i + 1 < x.size(); ++i) {
                int x1 = static_cast<int>((x[i] - x_min) * x_scale);
                int y1 = dot_height - 1 - static_cast<int>((y[i] - y_min) * y_scale);
                int x2 = static_cast<int>((x[i+1] - x_min) * x_scale);
                int y2 = dot_height - 1 - static_cast<int>((y[i+1] - y_min) * y_scale);

                // Simple line drawing
                int steps = std::max(std::abs(x2 - x1), std::abs(y2 - y1));
                if (steps > 0) {
                    for (int s = 0; s <= steps; ++s) {
                        int px = x1 + (x2 - x1) * s / steps;
                        int py = y1 + (y2 - y1) * s / steps;
                        px = std::clamp(px, 0, dot_width - 1);
                        py = std::clamp(py, 0, dot_height - 1);
                        dots[py][px] = true;
                    }
                }
            }
        }

        // Convert to braille
        std::ostringstream ss;

        if (!title.empty()) {
            ss << title << "\n";
        }

        for (int row = 0; row < height; ++row) {
            for (int col = 0; col < width; ++col) {
                // Braille dot pattern:
                // 0 3
                // 1 4
                // 2 5
                // 6 7
                int pattern = 0;
                int base_y = row * 4;
                int base_x = col * 2;

                if (base_y < dot_height && base_x < dot_width && dots[base_y][base_x]) pattern |= 0x01;
                if (base_y + 1 < dot_height && base_x < dot_width && dots[base_y + 1][base_x]) pattern |= 0x02;
                if (base_y + 2 < dot_height && base_x < dot_width && dots[base_y + 2][base_x]) pattern |= 0x04;
                if (base_y < dot_height && base_x + 1 < dot_width && dots[base_y][base_x + 1]) pattern |= 0x08;
                if (base_y + 1 < dot_height && base_x + 1 < dot_width && dots[base_y + 1][base_x + 1]) pattern |= 0x10;
                if (base_y + 2 < dot_height && base_x + 1 < dot_width && dots[base_y + 2][base_x + 1]) pattern |= 0x20;
                if (base_y + 3 < dot_height && base_x < dot_width && dots[base_y + 3][base_x]) pattern |= 0x40;
                if (base_y + 3 < dot_height && base_x + 1 < dot_width && dots[base_y + 3][base_x + 1]) pattern |= 0x80;

                // Unicode braille block starts at U+2800
                char32_t braille = 0x2800 + pattern;

                // Encode as UTF-8
                if (braille < 0x80) {
                    ss << static_cast<char>(braille);
                } else if (braille < 0x800) {
                    ss << static_cast<char>(0xC0 | (braille >> 6));
                    ss << static_cast<char>(0x80 | (braille & 0x3F));
                } else {
                    ss << static_cast<char>(0xE0 | (braille >> 12));
                    ss << static_cast<char>(0x80 | ((braille >> 6) & 0x3F));
                    ss << static_cast<char>(0x80 | (braille & 0x3F));
                }
            }
            ss << "\n";
        }

        return ss.str();
    }
};

// Factory function
std::unique_ptr<Command> create_plot_sixel() {
    return std::make_unique<PlotSixelCommand>();
}
