import random
import json

from dspy.teleprompt.teleprompt import Teleprompter

"""
TODO: The EnsembledProgram should actually imitate the structure of the individual programs (IF they are all compatible). This allows compiling with an ensemble program as a (singular) teacher. Basically the top majority-compatible trace will end up being used, if dspy.majority is the reduce_fn.
"""


class BestDemoEnsemble(Teleprompter):
    def __init__(self, *, reduce_fn=None, size=None, deterministic=False):
        """A common reduce_fn is dspy.majority."""

        assert deterministic is False, "TODO: Implement example hashing for deterministic ensemble."

        self.reduce_fn = reduce_fn
        self.size = size
        self.deterministic = deterministic

    def compile(self, programs, lm_judge):
        size = self.size
        reduce_fn = self.reduce_fn
        lm = lm_judge

        import dspy

        programs = random.sample(programs, size) if size else programs
        demos = {}
        for i, program in enumerate(programs):
            program_name = f"Program {i}"
            program_demos = {}
            for name, predictor in program.named_predictors():
                program_demos[name] = predictor.demos
            demos[program_name] = program_demos

        prompt = f'''
        Which program has a better set of demonstrations for in-context learning with an LLM?: {demos}. 
        Use the following criteria: difficulty, coverage/breadth, and semantic shift. 
        Moderate difficulty, high coverage, and low semantic shift between consecutive demos is good. 
        Respond with the program number and your justification for choosing this program in a json.
        '''
        output = lm(prompt)[0]
        start_idx = output.find("{")
        end_idx = output.rfind("}")
        output = json.loads(output[start_idx:end_idx+1])
        program_num = int(output["program"])
        justification = output['justification']

        class EnsembledProgram(dspy.Module):
            def __init__(self):
                super().__init__()
                self.programs = programs
                self.lm = lm_judge
                self.chosen_program = program_num
                self.justification = justification

            def forward(self, *args, **kwargs):
                output = programs[self.chosen_program](*args, **kwargs)

                if reduce_fn:
                    return reduce_fn(output)

                return output

        return EnsembledProgram()