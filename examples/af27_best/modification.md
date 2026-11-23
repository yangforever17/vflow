Added one bounded, reasoning-enabled fresh generation when the existing compiler-repair stage still leaves invalid RTL; accept it only after successful compilation.
The restart derives from the specification alone, avoiding continued edits to a broken implementation, and returns immediately on success.
Unlike round 18, both original repair opportunities remain and the restart uses reasoning; already-compiling paths are unchanged.
This targets residual compile/syntax failures while charging extra generation tokens only on the failed-compilation branch.
