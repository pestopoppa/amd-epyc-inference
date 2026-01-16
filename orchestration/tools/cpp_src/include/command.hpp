/**
 * @file command.hpp
 * @brief Base command interface for llama-math-tools
 *
 * All commands inherit from this base class and implement execute().
 */

#pragma once

#include <nlohmann/json.hpp>
#include <memory>
#include <string>
#include <chrono>

using json = nlohmann::json;

/**
 * @brief Base class for all math tool commands
 */
class Command {
public:
    virtual ~Command() = default;

    /**
     * @brief Execute the command with given parameters
     * @param params JSON object containing command parameters
     * @return JSON response with status and result
     */
    virtual json execute(const json& params) = 0;

    /**
     * @brief Get command name
     */
    virtual std::string name() const = 0;

    /**
     * @brief Get command description
     */
    virtual std::string description() const = 0;

protected:
    /**
     * @brief Create success response
     */
    json success(const json& result, const json& stats = json::object()) {
        json response;
        response["status"] = "success";
        response["result"] = result;
        if (!stats.empty()) {
            response["stats"] = stats;
        }
        response["error"] = nullptr;
        return response;
    }

    /**
     * @brief Create error response
     */
    json error(const std::string& message) {
        json response;
        response["status"] = "error";
        response["result"] = nullptr;
        response["error"] = message;
        return response;
    }

    /**
     * @brief Timer utility for stats
     */
    class Timer {
    public:
        Timer() : start_(std::chrono::high_resolution_clock::now()) {}

        double elapsed_ms() const {
            auto end = std::chrono::high_resolution_clock::now();
            return std::chrono::duration<double, std::milli>(end - start_).count();
        }

    private:
        std::chrono::time_point<std::chrono::high_resolution_clock> start_;
    };
};

/**
 * @brief Factory function type for creating commands
 */
using CommandFactory = std::unique_ptr<Command>(*)();
