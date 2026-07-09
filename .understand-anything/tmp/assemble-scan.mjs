import fs from "fs";
import path from "path";

const root = "C:/Users/Dharshan.K/OneDrive/Desktop/coral-";
const scan = JSON.parse(
  fs.readFileSync(path.join(root, ".understand-anything/tmp/ua-scan-files.json"), "utf8")
);
const imports = JSON.parse(
  fs.readFileSync(path.join(root, ".understand-anything/tmp/ua-import-map-output.json"), "utf8")
);

const description =
  "AI-driven unified data platform for oceanographic and biodiversity insights that integrates multi-source environmental datasets to analyze ocean conditions, detect ecological anomalies, and compute RM-NPI coastal risk metrics using machine learning and geospatial visualization. Note: this project has over 100 source files; consider scoping analysis to a subdirectory for faster results.";

const result = {
  name: "coral-ai",
  description,
  languages: Object.keys(scan.stats.byLanguage).sort(),
  frameworks: [
    "Docker",
    "Docker Compose",
    "FastAPI",
    "Pydantic",
    "React",
    "Streamlit",
    "Tailwind CSS",
    "Vite",
  ],
  files: scan.files,
  totalFiles: scan.totalFiles,
  filteredByIgnore: scan.filteredByIgnore,
  estimatedComplexity: scan.estimatedComplexity,
  importMap: imports.importMap,
};

const outDir = path.join(root, ".understand-anything/intermediate");
fs.mkdirSync(outDir, { recursive: true });
const outPath = path.join(outDir, "scan-result.json");
fs.writeFileSync(outPath, JSON.stringify(result, null, 2));

if (result.totalFiles !== result.files.length) {
  console.error("Mismatch:", result.totalFiles, result.files.length);
  process.exit(1);
}

console.log("written", outPath, result.totalFiles, "files");
