const cachedDeps = new Map();

function requireFromRoot(moduleName) {
  try {
    if (cachedDeps.has(moduleName)) return cachedDeps.get(moduleName);

    const modulePath = require.resolve(moduleName, { paths: [process.cwd()] });
    const resolvedModule = require(modulePath);
    cachedDeps.set(moduleName, resolvedModule);
    return resolvedModule;
  } catch (e) {
    throw new Error(
      `codeflash: Could not resolve '${moduleName}' from project root (${process.cwd()}). ` +
        `Ensure ${moduleName} is installed in your project: npm install ${moduleName}`
    );
  }
}

module.exports = {
  requireFromRoot,
};
