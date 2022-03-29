package executor

import (
	"fmt"
	"github.com/yanghaku/wasmer-gpu-go/wasmer"
	"io/ioutil"
	"log"
	"os"
	"strings"
	"sync"
	"time"
)

const (
	binDir  = "/bin/"
	dataDir = "/data/"
	runDir  = "/run/"
)

// defaultInstanceNum is the default number of instance
const defaultInstanceNum = 4

// maxInstanceNum is the max limit for running instance
const maxInstanceNum = 65535

// WasmFunctionRunner run a wasm function
type WasmFunctionRunner struct {
	ExecTimeout time.Duration
	LogPrefix   bool

	Process     string
	ProcessArgs []string
	DataAbsPath *string

	WasmRoot   string
	WasmModule *wasmer.Module
	WasmStore  *wasmer.Store

	freeFuncId chan int
	replicas   int
	mutex      sync.Mutex
}

// NewWasmFunctionRunner make a new WasmFunctionRunner and init for warm start
func NewWasmFunctionRunner(execTimeout time.Duration, prefixLogs bool,
	commandName string, commandArgs []string, wasmRoot string) (*WasmFunctionRunner, error) {

	runner := &WasmFunctionRunner{
		ExecTimeout: execTimeout,
		LogPrefix:   prefixLogs,
		Process:     commandName,
		ProcessArgs: commandArgs,
		WasmRoot:    wasmRoot,
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

	// config data path
	// if data path invalid, save the DataAbsPath as nil
	dataAbsPath := wasmRoot + dataDir + commandName + "/"
	exist, isDir, err = statDir(dataAbsPath)
	if err != nil {
		return nil, err
	}
	if exist && isDir {
		runner.DataAbsPath = &dataAbsPath
	} else {
		runner.DataAbsPath = nil
	}

	runner.replicas = defaultInstanceNum
	runner.freeFuncId = make(chan int, maxInstanceNum)
	for i := 0; i < runner.replicas; i++ {
		runner.freeFuncId <- i
	}

	return runner, nil
}

func (f *WasmFunctionRunner) Run(req FunctionRequest) error {
	funcId := <-f.freeFuncId
	defer func() {
		f.freeFuncId <- funcId
	}()

	// config the file system root directory for this Webassembly instance
	// we use the function id as the directory name
	wasmWorkDir := fmt.Sprintf("%d", funcId)
	exists, _, err := statDir(wasmWorkDir)
	if err != nil {
		return err
	}
	// remove the old work dir
	if exists {
		if err := os.RemoveAll(wasmWorkDir); err != nil {
			return err
		}
	}
	// create work dir for this instance
	if err := os.MkdirAll(wasmWorkDir, os.ModePerm); err != nil {
		return err
	}
	// create the link for all data file to run directory
	if f.DataAbsPath != nil {
		fileInfos, err := ioutil.ReadDir(*f.DataAbsPath)
		if err != nil {
			return err
		}
		for _, fi := range fileInfos {
			if err := os.Symlink(*f.DataAbsPath+fi.Name(), wasmWorkDir+"/"+fi.Name()); err != nil {
				return err
			}
		}
	}

	// running function
	err = f.runFunc(req, wasmWorkDir)

	// clean this work dir
	if err := os.RemoveAll(wasmWorkDir); err != nil {
		return err
	}

	return err
}

// runFunc instance a func and run it
func (f *WasmFunctionRunner) runFunc(req FunctionRequest, funcId string) error {
	log.Printf("process name =  %s", f.Process)
	log.Printf("process args = %s", f.ProcessArgs)
	log.Printf("running function Id = %s", funcId)
	log.Printf("exec Time out = %s", f.ExecTimeout.String())

	startTime := time.Now()

	wasiEnvBuilder := wasmer.NewWasiStateBuilder(f.Process).CaptureStdout().CaptureStderr()
	for _, arg := range f.ProcessArgs {
		wasiEnvBuilder.Argument(arg)
	}
	// map the root directory
	wasiEnvBuilder.MapDirectory("/", funcId)
	wasiEnvBuilder.Environment("PWD", "/")
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
		// try capture the stderr
		go wasmLogging(f.Process+":"+funcId, wasiEnv.ReadStderr(), f.LogPrefix)
		return err
	}

	_, err = req.OutputWriter.Write(wasiEnv.ReadStdout())
	if err != nil {
		return err
	}

	duringTime := time.Since(startTime)
	log.Printf("Took %v us ( %v ms )", duringTime.Microseconds(), duringTime.Milliseconds())

	// capture the stderr for function
	go wasmLogging(f.Process+":"+funcId, wasiEnv.ReadStderr(), f.LogPrefix)

	return nil
}

// ReadScale return the replicas of functions
func (f *WasmFunctionRunner) ReadScale() int {
	f.mutex.Lock()
	defer f.mutex.Unlock()
	return f.replicas
}

// ScaleFunc scale the replicas
func (f *WasmFunctionRunner) ScaleFunc(replicas int) error {
	f.mutex.Lock()
	defer f.mutex.Unlock()
	if replicas == f.replicas {
		return nil
	}
	if replicas > f.replicas {
		if replicas > maxInstanceNum {
			return fmt.Errorf("the replicas cannot greater than maxInstanceNum(%d)", maxInstanceNum)
		}
		for i := f.replicas; i < replicas; i++ {
			f.freeFuncId <- i
		}
		f.replicas = replicas
	}
	// todo: shrink for replicas
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

// wasmLogging log the stderr for functions to os.Stderr
func wasmLogging(name string, bytes []byte, logPrefix bool) {
	log.Printf("Started logging: %s from function.", name)

	var logger *log.Logger
	if logPrefix {
		logger = log.New(os.Stderr, log.Prefix(), log.Flags())
	} else {
		logger = log.New(os.Stderr, "", 0)
	}

	logs := strings.Split(string(bytes), "\n")
	for _, s := range logs {
		if logPrefix {
			logger.Printf("%s: %s", name, s)
		} else {
			logger.Printf(s)
		}
	}
}
