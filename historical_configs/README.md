# Historical training configuration references

These are previous actual runs: Teacher MAX/OFF, CI2000/CI8197 and 772 M1/M2.
They are distinct from the newly prepared CalibForge High CF1000 native recipe.
Each JSON includes the original LF arguments, DeepSpeed configuration, historical
entry, source artifact path, available serialized runtime fields, and identity.
Runtime fields distinguish changes made after parsing the original arguments.
These older runs used the project preprocessing/wrappers; the new clean native
entry does not become historically equivalent merely by copying these flags.
Teacher OFF denotes the teacher's rollout mode; student thinking remained enabled.
No checkpoint, optimizer tensor, private environment or credential is included.
