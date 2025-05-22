---
sidebar_position: 2
---

# Optimize your entire codebase

Codeflash can optimize your entire codebase by analyzing all the functions in your project and generating optimized versions of them.
It iterates through all the functions in your codebase and optimizes them one by one.

To optimize your entire codebase, run the following command in your project directory:

```bash
codeflash --all
```

This requires the Codeflash GitHub App to be installed in your repository.

This is a powerful feature that can help you optimize your entire codebase in one go.
Since it runs on all the functions in your codebase, it can take some time to complete, please be patient.
As this runs you will see Codeflash opening pull requests for each function it successfully optimizes.

## Important considerations
- **Dedicated Optimization Machine:** Optimizing the entire codebase may require considerable time—up to one day. It's recommended to allocate a dedicated machine specifically for this long-running optimization task.

- **Minimize Background Processes:** To achieve optimal results, avoid running other processes on the optimization machine. Additional processes can introduce noise into Codeflash's runtime measurements, reducing the quality of the optimizations. Although Codeflash tolerates some runtime fluctuations, minimizing noise ensures the highest optimization quality.

- **Checkpoint and Recovery:** Codeflash automatically creates checkpoints as it identifies optimizations. If the optimization process is interrupted or encounters issues, you can resume the process by re-running `codeflash --all`. The command will prompt you to continue from the most recent checkpoint.
