package executor

import (
	"github.com/yanghaku/wasmer-gpu-go/wasmer"
	"io/ioutil"
	"log"
	"os"
	"time"
)

const (
	binDir  = "/bin/"
	dataDir = "/data/"
	runDir  = "/run/"
)

// WasmFunctionRunner run a wasm function
type WasmFunctionRunner struct {
	ExecTimeout   time.Duration
	LogPrefix     bool
	LogBufferSize int

	Process     string
	ProcessArgs []string

	WasmRoot   string
	WasmModule *wasmer.Module
	WasmStore  *wasmer.Store
}

// NewWasmFunctionRunner make a new WasmFunctionRunner and init for warm start
func NewWasmFunctionRunner(execTimeout time.Duration, prefixLogs bool, logBufferSize int,
	commandName string, arguments []string, wasmRoot string) (*WasmFunctionRunner, error) {

	runner := &WasmFunctionRunner{
		ExecTimeout:   execTimeout,
		LogPrefix:     prefixLogs,
		LogBufferSize: logBufferSize,
		Process:       commandName,
		ProcessArgs:   arguments,
		WasmRoot:      wasmRoot,
	}

	// default the wasm file store in WasmRoot/bin
	wasmBytes, err := ioutil.ReadFile(wasmRoot + binDir + commandName)
	if err != nil {
		return nil, err
	}

	store := wasmer.NewStore(wasmer.NewEngine())
	module, err := wasmer.NewModule(store, wasmBytes)
	if err != nil {
		return nil, err
	}

	runner.WasmStore = store
	runner.WasmModule = module

	// check work directory
	runAbsPath := wasmRoot + runDir + commandName
	exist, isDir, err := statDir(runAbsPath)
	if err != nil {
		return nil, err
	}
	if exist {
		if err := os.RemoveAll(runAbsPath); err != nil {
			return nil, err
		}
	}

	if err := os.MkdirAll(runAbsPath, os.ModePerm); err != nil {
		return nil, err
	}

	// change to work directory
	if err := os.Chdir(runAbsPath); err != nil {
		log.Println(err)
		os.Exit(1)
	}

	// check data path
	dataAbsPath := wasmRoot + dataDir + commandName + "/"
	exist, isDir, err = statDir(dataAbsPath)
	if err != nil {
		return nil, err
	}
	if exist && isDir { // create the link for all data file to run directory
		fileInfos, err := ioutil.ReadDir(dataAbsPath)
		if err != nil {
			return nil, err
		}
		for _, fi := range fileInfos {
			if err := os.Symlink(dataAbsPath+fi.Name(), fi.Name()); err != nil {
				return nil, err
			}
		}
	}

	return runner, nil
}

func (f *WasmFunctionRunner) Run(req FunctionRequest) error {
	log.Printf("process name =  %s", f.Process)
	log.Printf("process args = %s", f.ProcessArgs)
	log.Printf("exec Time out = %s", f.ExecTimeout.String())

	startTime := time.Now()

	wasiEnvBuilder := wasmer.NewWasiStateBuilder(f.Process).CaptureStdout().CaptureStderr()
	for _, arg := range f.ProcessArgs {
		wasiEnvBuilder.Argument(arg)
	}
	// todo: resolve the environment variable

	wasiEnv, err := wasiEnvBuilder.Finalize()
	if err != nil {
		log.Println(err)
		return err
	}

	importObject, err := wasiEnv.GenerateImportObject(f.WasmStore, f.WasmModule)
	if err != nil {
		log.Println(err)
		return err
	}

	cudaEnv := wasmer.NewCudaEnvironment()
	err = cudaEnv.AddImportObject(f.WasmStore, importObject)
	if err != nil {
		log.Println(err)
		return err
	}

	instance, err := wasmer.NewInstance(f.WasmModule, importObject)
	if err != nil {
		log.Println(err)
		return err
	}

	start, err := instance.Exports.GetWasiStartFunction()
	if err != nil {
		log.Println(err)
		return err
	}

	if req.InputReader != nil {
		// todo: read the stdin
		defer req.InputReader.Close()
	}

	// execute time out
	var timer *time.Timer
	if f.ExecTimeout > 0 {
		timer = time.NewTimer(f.ExecTimeout)
		go func() {
			<-timer.C

			log.Printf("Function was killed by ExecTimeout: %s\n", f.ExecTimeout.String())

			instance.Close()
		}()
	}

	if timer != nil {
		defer timer.Stop()
	}

	_, err = start()
	if err != nil {
		return err
	}
	req.OutputWriter.Write(wasiEnv.ReadStdout())

	log.Printf("stderr = %s", string(wasiEnv.ReadStderr()))

	duringTime := time.Since(startTime)
	log.Printf("Took %v us ( %v ms )", duringTime.Microseconds(), duringTime.Milliseconds())

	return nil
}

// statDir return (file if exists, file is dir, error)
func statDir(path string) (bool, bool, error) {
	fi, err := os.Stat(path)
	if err == nil {
		return true, fi.IsDir(), nil
	}
	if os.IsNotExist(err) {
		return false, false, nil
	}
	return false, false, err
}
