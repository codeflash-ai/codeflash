/**
 * Codeflash JUnit XML Reporter for Jest.
 *
 * Minimal reporter that outputs JUnit XML in the format expected by
 * codeflash's Python parser. Replaces the external jest-junit dependency.
 *
 * Configuration via environment variables (same as jest-junit):
 *   JEST_JUNIT_OUTPUT_FILE  — absolute path for the XML file (required)
 *   JEST_JUNIT_CLASSNAME    — template for classname ("{filepath}" supported)
 *   JEST_JUNIT_SUITE_NAME   — template for suite name ("{filepath}" supported)
 *   JEST_JUNIT_ADD_FILE_ATTRIBUTE — "true" to add file= on <testcase>
 *   JEST_JUNIT_INCLUDE_CONSOLE_OUTPUT — "true" to include console.log in <system-out>
 */

"use strict";

const fs = require("fs");
const path = require("path");

function escapeXml(str) {
  if (!str) return "";
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&apos;");
}

function escapeXmlContent(str) {
  if (!str) return "";
  return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function formatTemplate(template, values) {
  if (!template) return "";
  let result = template;
  for (const [key, val] of Object.entries(values)) {
    result = result.replace(new RegExp(`\\{${key}\\}`, "g"), val || "");
  }
  return result;
}

class CodeflashJestReporter {
  constructor(globalConfig, _reporterOptions) {
    this._globalConfig = globalConfig;
    this._outputFile = process.env.JEST_JUNIT_OUTPUT_FILE || "jest-results.xml";
    this._classnameTemplate = process.env.JEST_JUNIT_CLASSNAME || "{classname}";
    this._suiteNameTemplate = process.env.JEST_JUNIT_SUITE_NAME || "{filepath}";
    this._addFileAttribute =
      process.env.JEST_JUNIT_ADD_FILE_ATTRIBUTE === "true";
    this._includeConsoleOutput =
      process.env.JEST_JUNIT_INCLUDE_CONSOLE_OUTPUT === "true";
    // Capture buffered console output per test file
    this._consoleBuffers = new Map();
  }

  // Called by Jest when a test suite starts — we just note it
  onTestStart(_test) {}

  // Called by Jest with console output for a test file
  onTestFileResult(_test, testResult, _aggregatedResult) {
    if (
      this._includeConsoleOutput &&
      testResult.console &&
      testResult.console.length > 0
    ) {
      const messages = testResult.console
        .map((entry) => {
          const prefix =
            entry.type === "error"
              ? "console.error"
              : entry.type === "warn"
                ? "console.warn"
                : "console.log";
          return `${prefix}\n  ${entry.message}`;
        })
        .join("\n\n");
      this._consoleBuffers.set(testResult.testFilePath, messages);
    }
  }

  onRunComplete(_testContexts, results) {
    const suites = [];
    let totalTests = 0;
    let totalFailures = 0;
    let totalErrors = 0;
    let totalTime = 0;

    for (const suiteResult of results.testResults) {
      const filePath = suiteResult.testFilePath || "";
      const relativePath = this._globalConfig.rootDir
        ? path.relative(this._globalConfig.rootDir, filePath)
        : filePath;

      const templateVars = {
        filepath: filePath,
        filename: path.basename(filePath),
        classname: relativePath.replace(/\//g, ".").replace(/\.[^.]+$/, ""),
        title: "",
        displayName: suiteResult.displayName || "",
      };

      const suiteName = formatTemplate(
        this._suiteNameTemplate,
        templateVars
      );

      const testcases = [];
      let suiteFailures = 0;
      let suiteErrors = 0;
      let suiteTime = 0;

      for (const testResult of suiteResult.testResults) {
        const duration = (testResult.duration || 0) / 1000; // ms → seconds
        suiteTime += duration;

        const tcTemplateVars = {
          ...templateVars,
          title: testResult.fullName || testResult.title || "",
        };

        const classname = formatTemplate(
          this._classnameTemplate,
          tcTemplateVars
        );

        let tcXml = `    <testcase classname="${escapeXml(classname)}" name="${escapeXml(
          testResult.fullName || testResult.title
        )}" time="${duration.toFixed(3)}"`;

        if (this._addFileAttribute) {
          tcXml += ` file="${escapeXml(filePath)}"`;
        }

        if (
          testResult.status === "failed" &&
          testResult.failureMessages &&
          testResult.failureMessages.length > 0
        ) {
          suiteFailures++;
          const failureText = testResult.failureMessages.join("\n");
          tcXml += `>\n      <failure message="${escapeXml(
            failureText.split("\n")[0]
          )}"><![CDATA[${failureText}]]></failure>\n    </testcase>`;
        } else if (testResult.status === "pending") {
          tcXml += `>\n      <skipped/>\n    </testcase>`;
        } else {
          tcXml += "/>";
        }

        testcases.push(tcXml);
      }

      totalTests += suiteResult.testResults.length;
      totalFailures += suiteFailures;
      totalErrors += suiteErrors;
      totalTime += suiteTime;

      // Build suite XML
      let suiteXml = `  <testsuite name="${escapeXml(suiteName)}" tests="${
        suiteResult.testResults.length
      }" errors="${suiteErrors}" failures="${suiteFailures}" skipped="${
        suiteResult.testResults.filter((t) => t.status === "pending").length
      }" timestamp="${new Date().toISOString()}" time="${suiteTime.toFixed(3)}"`;

      if (this._addFileAttribute) {
        suiteXml += ` file="${escapeXml(filePath)}"`;
      }

      suiteXml += ">\n";
      suiteXml += testcases.join("\n") + "\n";

      // Add console output as system-out (at suite level, matching jest-junit format)
      if (this._includeConsoleOutput) {
        const consoleOutput =
          this._consoleBuffers.get(suiteResult.testFilePath) || "";
        if (consoleOutput) {
          suiteXml += `    <system-out><![CDATA[${consoleOutput}]]></system-out>\n`;
        }
      }

      suiteXml += "  </testsuite>";
      suites.push(suiteXml);
    }

    const xml = [
      '<?xml version="1.0" encoding="UTF-8"?>',
      `<testsuites name="jest tests" tests="${totalTests}" failures="${totalFailures}" errors="${totalErrors}" time="${totalTime.toFixed(3)}">`,
      ...suites,
      "</testsuites>",
    ].join("\n");

    // Ensure output directory exists
    const outputDir = path.dirname(this._outputFile);
    if (!fs.existsSync(outputDir)) {
      fs.mkdirSync(outputDir, { recursive: true });
    }

    fs.writeFileSync(this._outputFile, xml, "utf8");
  }
}

module.exports = CodeflashJestReporter;
