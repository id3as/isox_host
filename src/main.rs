use isox_comms::{
    set_non_blocking, FromExtensionMsg, IsoxTermDeserializationContext, IsoxTermDeserialize,
    IsoxTermSerializationContext, IsoxTermSerialize, ReadMessageContext, ReadMessageResult, Term,
    ToExtensionMsg,
};
use libisox_host::{log_to_file, ptr_to_arc, Env, IsoxFunctionTable, HOST_GLOBALS};
use mio::unix::SourceFd;
use mio::{Events, Interest, Poll, Token};
use std::env;
use std::ffi::c_void;
use std::io::{ErrorKind, Read};
use std::os::fd::{AsRawFd, OwnedFd};
use std::os::unix::io::FromRawFd;
use std::rc::Rc;
use std::sync::Arc;

fn main() -> nix::Result<()> {
    const SOCKET_TOKEN: Token = Token(0);
    const STDIN_TOKEN: Token = Token(1);

    let args: Vec<String> = env::args().collect();
    let library = &args[1];

    let socket_fd_str = env::var("CHILD_SOCKET_FD").expect("Missing socket FD");
    let socket_fd = socket_fd_str.parse::<i32>().expect("Invalid FD");
    let socket = unsafe { OwnedFd::from_raw_fd(socket_fd) };

    set_non_blocking(0);
    set_non_blocking(socket_fd);

    unsafe { HOST_GLOBALS.set_write_socket(&socket) };

    // Load the library
    let library_with_ext = append_extension(library);
    let lib = unsafe {
        libloading::Library::new(library_with_ext.clone()).expect("Failed to load library")
    };
    let function_table = unsafe {
        let get_function_table: libloading::Symbol<
            unsafe extern "C" fn() -> &'static IsoxFunctionTable,
        > = lib.get(b"isox_init").unwrap();

        get_function_table()
    };

    eprintln!("ISOX Extension Library loaded");

    log_to_file!("Loaded lib {}", library_with_ext);

    // And go into our poll loop
    let mut poll = Poll::new().expect("fail to create poll");

    poll.registry()
        .register(
            &mut SourceFd(&socket.as_raw_fd()),
            SOCKET_TOKEN,
            Interest::READABLE.add(Interest::WRITABLE),
        )
        .expect("failed to register");

    poll.registry()
        .register(
            &mut SourceFd(&std::io::stdin().as_raw_fd()),
            STDIN_TOKEN,
            Interest::READABLE,
        )
        .expect("failed to register");

    let mut events = Events::with_capacity(128);
    let mut read_message_ctx = ReadMessageContext::new(&socket);
    let mut keep_running = true;
    let mut loaded = false;

    while keep_running {
        // poll.poll(&mut events, None).expect("poll failed");
        match poll.poll(&mut events, None) {
            Ok(_) => {
                for event in &events {
                    match event.token() {
                        SOCKET_TOKEN => {
                            if event.is_readable() {
                                loop {
                                    match read_message_ctx.read_message() {
                                        ReadMessageResult::WouldBlock => {
                                            break;
                                        }
                                        ReadMessageResult::Error(e) => {
                                            log_to_file!("Error doing recvmsg {}", e)
                                        }
                                        ReadMessageResult::Continue => (),
                                        ReadMessageResult::Ok((data, fds)) => {
                                            let msg = ToExtensionMsg::isox_term_deserialize(
                                                &mut IsoxTermDeserializationContext::new(
                                                    &data, fds,
                                                ),
                                            );

                                            match msg {
                                                ToExtensionMsg::InitCommand { arguments } => {
                                                    if loaded {
                                                        panic!("Multiple InitCommands received");
                                                    } else {
                                                        keep_running =
                                                            call_load(function_table, arguments);
                                                        loaded = true;
                                                    }
                                                }
                                                ToExtensionMsg::QueryCommand {
                                                    command_id,
                                                    arguments,
                                                } => {
                                                    keep_running = call_query(
                                                        function_table,
                                                        command_id,
                                                        arguments,
                                                    )
                                                }
                                                ToExtensionMsg::CreateInstanceCommand {
                                                    command_id,
                                                    arguments,
                                                } => {
                                                    keep_running = call_create_instance(
                                                        function_table,
                                                        command_id,
                                                        arguments,
                                                    )
                                                }
                                                ToExtensionMsg::QueryInstanceCommand {
                                                    command_id,
                                                    instance,
                                                    arguments,
                                                } => {
                                                    keep_running = call_query_instance(
                                                        function_table,
                                                        command_id,
                                                        instance,
                                                        arguments,
                                                    )
                                                }
                                                ToExtensionMsg::UpdateInstanceCommand {
                                                    command_id,
                                                    instance,
                                                    arguments,
                                                } => {
                                                    keep_running = call_update_instance(
                                                        function_table,
                                                        command_id,
                                                        instance,
                                                        arguments,
                                                    )
                                                }
                                                ToExtensionMsg::DestroyInstanceCommand {
                                                    command_id,
                                                    instance,
                                                } => {
                                                    keep_running = call_destroy_instance(
                                                        function_table,
                                                        command_id,
                                                        instance,
                                                    )
                                                }
                                                ToExtensionMsg::ResourceDestructorCommand {
                                                    resource_type_id,
                                                    resource_data,
                                                } => {
                                                    let resource_record = unsafe {
                                                        HOST_GLOBALS
                                                            .get_resource_record(resource_type_id)
                                                    };
                                                    let mut env = Env::new();
                                                    resource_record.call_destructor(
                                                        &mut env,
                                                        resource_data.as_ptr() as *mut c_void,
                                                    );
                                                }
                                            };
                                        }
                                    }
                                }
                            }
                        }
                        STDIN_TOKEN => loop {
                            let mut buf = vec![0; 1024];
                            match std::io::stdin().read(&mut buf) {
                                Ok(0) => {
                                    keep_running = false;
                                    break;
                                }
                                Ok(_) => (),
                                Err(err) if err.kind() == ErrorKind::WouldBlock => (),
                                Err(_) => {
                                    keep_running = false;
                                    break;
                                }
                            }
                        },
                        _ => log_to_file!("Unknown event"),
                    }
                }
            }
            Err(_e) => (),
        }
    }
    Ok(())
    /*
    // todo
    (function_table.unload)(globals);
    */
}

fn create_env(term: &Arc<Term>) -> (*mut Env, Arc<Env>) {
    let mut env = Arc::new(Env::new_with_term(term.clone()));
    let env_ptr = Arc::into_raw(env);
    unsafe { (env_ptr as *mut Env, Arc::from_raw(env_ptr)) }
}

fn call_load(function_table: &IsoxFunctionTable, arguments: Arc<Term>) -> bool {
    let (env_ptr, env) = create_env(&arguments);
    let args = Arc::as_ptr(&arguments);
    let mut priv_data: *const c_void = std::ptr::null();
    let error_term_ptr =
        (function_table.load)(function_table.context, env_ptr, args, &mut priv_data);
    if error_term_ptr.is_null() {
        unsafe { HOST_GLOBALS.set_priv_data(priv_data) };
        true
    } else {
        let error_term = ptr_to_arc(error_term_ptr);
        let response = FromExtensionMsg::PluginFailed {
            term: error_term.clone(),
        };
        let mut ctx = IsoxTermSerializationContext::new();
        response.isox_term_serialize(&mut ctx);
        unsafe { HOST_GLOBALS.send_message(ctx) };
        false
    }
}

fn call_query(function_table: &IsoxFunctionTable, command_id: u64, arguments: Arc<Term>) -> bool {
    let (env_ptr, env) = create_env(&arguments);
    let args = Arc::as_ptr(&arguments);
    let query_result_ptr = (function_table.query)(function_table.context, env_ptr, args);
    let query_result = ptr_to_arc(query_result_ptr);
    let response = FromExtensionMsg::QueryResponse {
        command_id,
        term: query_result.clone(),
    };
    let mut ctx = IsoxTermSerializationContext::new();
    response.isox_term_serialize(&mut ctx);
    unsafe { HOST_GLOBALS.send_message(ctx) }
}

fn call_create_instance(
    function_table: &IsoxFunctionTable,
    command_id: u64,
    arguments: Arc<Term>,
) -> bool {
    let (env_ptr, env) = create_env(&arguments);
    let query_result_ptr =
        (function_table.create_instance)(function_table.context, env_ptr, Arc::as_ptr(&arguments));
    let query_result = ptr_to_arc(query_result_ptr);
    let response = FromExtensionMsg::CreateInstanceResponse {
        command_id,
        instance: query_result.clone(),
    };
    let mut ctx = IsoxTermSerializationContext::new();
    response.isox_term_serialize(&mut ctx);
    unsafe { HOST_GLOBALS.send_message(ctx) }
}

fn call_query_instance(
    function_table: &IsoxFunctionTable,
    command_id: u64,
    instance_id: Arc<Term>,
    arguments: Arc<Term>,
) -> bool {
    let (env_ptr, env) = create_env(&arguments);
    let query_result_ptr = (function_table.query_instance)(
        function_table.context,
        env_ptr,
        Arc::as_ptr(&instance_id),
        Arc::as_ptr(&arguments),
    );
    let query_result = ptr_to_arc(query_result_ptr);
    let response = FromExtensionMsg::QueryInstanceResponse {
        command_id,
        term: query_result.clone(),
    };
    let mut ctx = IsoxTermSerializationContext::new();
    response.isox_term_serialize(&mut ctx);
    unsafe { HOST_GLOBALS.send_message(ctx) }
}

fn call_update_instance(
    function_table: &IsoxFunctionTable,
    command_id: u64,
    instance_id: Arc<Term>,
    arguments: Arc<Term>,
) -> bool {
    let (env_ptr, env) = create_env(&arguments);
    let query_result_ptr = (function_table.update_instance)(
        function_table.context,
        env_ptr,
        Arc::as_ptr(&instance_id),
        Arc::as_ptr(&arguments),
    );

    let query_result = ptr_to_arc(query_result_ptr);
    let response = FromExtensionMsg::UpdateInstanceResponse {
        command_id,
        term: query_result.clone(),
    };
    let mut ctx = IsoxTermSerializationContext::new();

    response.isox_term_serialize(&mut ctx);

    unsafe { HOST_GLOBALS.send_message(ctx) }
}

fn call_destroy_instance(
    function_table: &IsoxFunctionTable,
    command_id: u64,
    instance_id: Arc<Term>,
) -> bool {
    let env = Arc::new(Env::new());
    let env_ptr = Arc::into_raw(env);
    let query_result_ptr = (function_table.destroy_instance)(
        function_table.context,
        env_ptr as *mut Env,
        Arc::as_ptr(&instance_id),
    );
    let query_result = ptr_to_arc(query_result_ptr);
    let response = FromExtensionMsg::DestroyInstanceResponse {
        command_id,
        term: query_result.clone(),
    };
    let mut ctx = IsoxTermSerializationContext::new();
    response.isox_term_serialize(&mut ctx);
    unsafe { HOST_GLOBALS.send_message(ctx) }
}

#[cfg(target_os = "macos")]
fn append_extension(lib_name: &String) -> String {
    format!("{}.dylib", lib_name)
}

#[cfg(target_os = "linux")]
fn append_extension(lib_name: &String) -> String {
    format!("{}.so", lib_name)
}
