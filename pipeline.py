import copy


class Pipeline:
    def __init__(self, name):
        self.name = name
        self.data = {}
        self.pipeline = []

    def generate(self, name, f, **args):
        self.pipeline.append(
            {'name': name, 'f': f, 'args': args, 'history': False})
        return self

    def transform(self, src, name, f, store_history=False, **args):
        self.pipeline.append(
            {'name': name, 'src': src, 'f': f, 'args': args, 'history': store_history})
        return self

    def val(self, name):
        if self.has(name):
            return self.data[name][0]
        else:
            return None

    def history(self, name):
        if self.has(name) and self.data[name][1]:
            return self.data[name][2]
        else:
            return []

    def has(self, name):
        return name in self.data

    # все преобразования выполняются на указанном пайплайне
    def apply(self, pipeline, **args):
        if type(pipeline) == list:
            res = []
            for pipeline in pipeline:
                res.append(self.apply(pipeline, **args))
            return res

        for stage in self.pipeline:
            targs = self._map_args(stage, args)

            # если стадия отключена
            if 'p_enable' in stage['args']:
                if (not 'p_enable' in targs) or targs['p_enable'] == False:
                    continue

            repeat = 1
            if ('p_repeat' in targs):
                repeat = targs['p_repeat']
                if(type(repeat) == tuple):
                    repeat = [repeat]
                
            if(type(repeat) == list):
                for i in range(len(pipeline.val(repeat[0][0]))):
                    for arr, var in repeat:
                        pipeline.data[var] = (pipeline.val(arr)[i], False, [])
                    self._apply_pipeline(pipeline, stage, args, targs)
            else:
                for _ in range(repeat):
                    self._apply_pipeline(pipeline, stage, args, targs)

        return pipeline
    
    def apply_copy(self, name, pipeline, **args):
        if(type(pipeline) == list):
            return self.apply(Pipeline.copy_list(pipeline, name, True), **args)
        else:
            return self.apply(pipeline.copy(name, True), **args)

    # выполнить все преобразования на самом себе
    def process(self, **args):
        return self.apply(self, **args)
    
    def process_copy(self, name, **args):
        return self.apply_copy(name, self, **args)

    def copy(self, name=None, copy_data=False):
        if name == None:
            name = self.name
        pcopy = Pipeline(name)
        pcopy.pipeline = copy.deepcopy(self.pipeline)
        if (copy_data):
            pcopy.data = copy.deepcopy(self.data)
        return pcopy

    def copy_list(pipelines, prefix, copy_data=False):
        res = []
        for pipe in pipelines:
            res.append(pipe.copy(prefix + "_" + pipe.name, copy_data))
        return res

    def subpipeline(self, pipe, **args):
        self.pipeline.append({"pipe": pipe, "args": args})

    # объединяем несколько пайплайнов в один
    def compose(pipes, name = ""):
        pcopy = Pipeline(name)

        for pipe in pipes:
            if (type(pipe) == tuple):
                pcopy.pipeline.append({"pipe": pipe[0], "args": pipe[1]})
            else:
                pcopy.pipeline.append({"pipe": pipe, "args": {}})

        return pcopy
    
    def compose_apply(name, pipe, pipes, **args):
        return Pipeline.compose(pipes).apply(pipe.copy(name, True), **args)

    # повторяем один и тотже пайплайн несколько раз
    def repeat(pipe, times, copy_data=False):
        pcopy = Pipeline(pipe.name + "_repeat_" + str(times))
        if copy_data:
            pcopy.data = copy.deepcopy(pipe.data)

        for _ in range(times):
            pcopy.pipeline += pipe.pipeline

        return pcopy
    
    def _apply_pipeline(self, pipeline, stage, args, targs):
        if 'pipe' in stage:
            stage['pipe'].apply(pipeline, **args)
        else:
            keywords = ['p_enable', 'p_repeat']
            clean_args = {k: v for k, v in targs.items() if k not in keywords}
            self._process_stage(pipeline, stage, clean_args)

    def _process_stage(self, pipeline, stage, args):
        # если выполняется преобразование
        values = []
        if 'src' in stage:
            src = stage['src']
            if type(src) != list:
                src = [src]

            for s in src:
                val = self.val(s)
                if val is None:
                    val = pipeline.val(s)
                if val is None:
                    raise ValueError('value key \"' + s + '\" not found')
                values.append(val)

        new_value = stage['f'](*values, **args)
        self._assign_result(pipeline, stage, new_value)

        if (stage['history']):
            self._append_result(pipeline, stage, new_value)

    # подставляем значения в аргументы
    def _map_args(self, stage, args):
        res = {}
        for k, v in stage['args'].items():
            if (type(v) == str and v.startswith('_')):
                if v in args:
                    res[k] = args[v]
            else:
                res[k] = v
        return res

    def _has_history(self, pipeline, name, history):
        if pipeline.has(name):
            has, his = pipeline.data[name][1], pipeline.data[name][2]
            return history or has, his 
        else:
            return False, []
        
    def _assign_result(self, pipeline, stage, res):
        if (type(stage['name']) == list):
            for i, n in enumerate(stage['name']):
                has, history = self._has_history(pipeline, n, stage['history'])
                pipeline.data[n] = (res[i], has, history)
        else:
            has, history = self._has_history(pipeline, stage['name'], stage['history'])
            pipeline.data[stage['name']] = (res, has, history)

    def _append_result(self, pipeline, stage, res):
        if (type(stage['name']) == list):
            for i, n in enumerate(stage['name']):
                pipeline.data[n][2].append(res[i])
        else:
            pipeline.data[stage['name']][2].append(res)