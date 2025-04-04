use std::{
    borrow::{Borrow, BorrowMut},
    collections::HashMap,
    ffi::{c_char, c_uint, CStr, CString},
    hash::{DefaultHasher, Hash, Hasher},
    os::fd::OwnedFd,
    sync::Arc,
};

use byteorder::WriteBytesExt;
use isox_comms::{
    mutex_resource, send_message, CommsSocket, FromExtensionMsg, GuardedCommsSocket,
    IsoxTermSerializationContext, IsoxTermSerialize, LogLevel, RawPointer, ResourceRefData,
    ShmSegment, Term, TermValue,
};
use libc::c_void;
use once_cell::sync::Lazy;

pub type ResourceDestructorFn =
    extern "C" fn(context: *const c_void, env: *mut Env, obj: *mut c_void);

pub struct ResourceType {
    pub resource_type_id: u64,
}

#[derive(Clone)]
pub struct ResourceRecord {
    pub resource_type: Arc<ResourceType>,
    destructor: ResourceDestructorFn,
    context: *const c_void,
}

impl ResourceRecord {
    pub fn call_destructor(&self, env: *mut Env, obj: *mut c_void) {
        (self.destructor)(self.context, env, obj);
    }
}

pub struct HostGlobals_ {
    priv_data: *const c_void,
    next_resource_type_id: u64,
    resource_records: HashMap<u64, ResourceRecord>,
    write_socket: Option<GuardedCommsSocket>,
}

mutex_resource!(HostGlobals_, HostGlobals);

impl HostGlobals_ {
    fn new() -> HostGlobals_ {
        HostGlobals_ {
            priv_data: std::ptr::null(),
            next_resource_type_id: 1,
            resource_records: HashMap::new(),
            write_socket: None,
        }
    }
}

pub static mut HOST_GLOBALS: Lazy<HostGlobals> =
    Lazy::new(|| HostGlobals::new(HostGlobals_::new()));

impl HostGlobals {
    pub fn set_write_socket(&self, socket: &OwnedFd) {
        let guarded_socket = GuardedCommsSocket::new(CommsSocket {
            socket: socket.try_clone().expect("Failed to clone"),
        });
        let mut globals = self.lock();
        globals.write_socket = Some(guarded_socket);
    }
    pub fn priv_data(&self) -> *mut c_void {
        self.lock().priv_data as *mut c_void
    }
    pub fn set_priv_data(&self, priv_data: *const c_void) {
        self.lock().priv_data = priv_data;
    }
    pub fn create_resource_type(
        &self,
        destructor: ResourceDestructorFn,
        context: *const c_void,
    ) -> Arc<ResourceType> {
        let mut globals = self.lock();
        let resource_type_id = globals.next_resource_type_id;
        let resource_type = Arc::new(ResourceType { resource_type_id });
        globals.next_resource_type_id += 1;

        globals.resource_records.insert(
            resource_type_id,
            ResourceRecord {
                resource_type: resource_type.clone(),
                destructor,
                context,
            },
        );
        resource_type
    }
    pub fn get_resource_record(&self, id: u64) -> ResourceRecord {
        self.lock()
            .resource_records
            .get(&id)
            .expect("Invalid resource id")
            .clone()
    }
    pub fn send_message(&self, serialised_message: IsoxTermSerializationContext) -> bool {
        let globals = self.lock();
        match &globals.write_socket {
            Some(write_socket) => {
                match send_message(&write_socket, serialised_message) {
                    Ok(_) => true,
                    Err(err) => {
                        eprintln!("SEND_MESSAGE - ERR {:?}", err);
                        log_to_file!("Send message returned error {:?}", err);
                        false
                    }
                };
                true
            }
            None => false,
        }
    }
}

pub struct Env {
    terms: Vec<Arc<Term>>,
}

// The Env holds a Vec of all the terms created in this env.  It exists
// to allow us to clean up the terms when the env is freed, saving the extension
// from having to keep track of all the terms they created to free them themselves.
// As such, when a term is just in the env with a pointer in the extension, then we
// want a refCount of 1 - i.e., the pointer in the extension *doesn't* have a ref count.
// This does mean that if the extension attempts to use a term after freeing an env,
// then it will crash.  Too bad.
impl Env {
    pub fn new() -> Self {
        Env { terms: Vec::new() }
    }
    pub fn new_with_term(term: Arc<Term>) -> Self {
        let mut terms = Vec::new();
        terms.push(term.clone());
        Env { terms }
    }

    fn add_term(&mut self, term: Term) -> *const Term {
        // We are adding a new term - so we wrap it in
        // an Rc (refCount now one), get a raw pointer (
        // initial Rc now consumed, refCount still one),
        // and then push a new Rc that results from from_raw,
        // so end result is one raw pointer, one Rc in our vec
        // with the refCount being one
        let raw = Arc::into_raw(Arc::new(term));
        self.terms.push(unsafe { Arc::from_raw(raw) });
        raw
    }

    pub fn add_existing_term(&mut self, term: Arc<Term>) {
        self.terms.push(term);
    }

    fn copy_term(&mut self, term: Arc<Term>) {
        self.terms.push(term);
    }
}

#[repr(C)]
pub enum TermType {
    Int = 1,
    Float = 2,
    String = 3,
    Pid = 4,
    Map = 5,
    List = 6,
    Binary = 7,
    Atom = 8,
    Tuple = 9,
    Resource = 10,
    ResourceRef = 11,
    Bool = 12,
}

impl TermType {
    fn from(value: u32) -> Self {
        match value {
            1 => TermType::Int,
            2 => TermType::Float,
            3 => TermType::String,
            4 => TermType::Pid,
            5 => TermType::Map,
            6 => TermType::List,
            7 => TermType::Binary,
            8 => TermType::Atom,
            9 => TermType::Tuple,
            10 => TermType::Resource,
            11 => TermType::ResourceRef,
            12 => TermType::Bool,
            _ => panic!("Unknown plugin term tag: {}", value),
        }
    }
}

#[no_mangle]
pub extern "C" fn isox_term_type(term: *const Term) -> TermType {
    unsafe {
        match (*term).value {
            TermValue::Bool(_) => TermType::Bool,
            TermValue::Int(_) => TermType::Int,
            TermValue::Float(_) => TermType::Float,
            TermValue::Pid(_) => TermType::Pid,
            TermValue::String(_) => TermType::String,
            TermValue::Map(_) => TermType::Map,
            TermValue::List(_) => TermType::List,
            TermValue::Shm(_) => TermType::Binary,
            TermValue::Atom(_) => TermType::Atom,
            TermValue::Tuple(_) => TermType::Tuple,
            TermValue::Resource(_) => TermType::Resource,
            TermValue::ResourceRef(_) => TermType::ResourceRef,
            TermValue::SubBinary(_, _, _) => TermType::Binary,
        }
    }
}

// Taking a ptr from the extension and recovering an Rc - but taking
// care to inc the refcount, since when this new Rc destructs, it will
// dec the refcountpub
pub fn ptr_to_arc<T>(ptr: *const T) -> Arc<T> {
    if ptr.is_null() {
        panic!("Attempt to cast null pointer to Term");
    }
    unsafe {
        Arc::increment_strong_count(ptr);
        Arc::from_raw(ptr)
    }
}

// As above, but with a bunch of term pointers, as would be used in tuples / lists
pub fn ptrs_to_arcs<T>(ptrs: *const *const T, num_terms: usize) -> Vec<Arc<T>> {
    if num_terms == 0 {
        Vec::new()
    } else {
        let term_ptrs = unsafe { std::slice::from_raw_parts(ptrs, num_terms) };
        term_ptrs
            .iter()
            .map(|&rc_ptr| ptr_to_arc(rc_ptr).clone())
            .collect()
    }
}

// Turn a pointer into a mutable reference
fn ptr_to_mut_ref<'a, T>(ptr: *mut T) -> &'a mut T {
    unsafe { &mut *ptr }
}

// Turn a pointer into a const reference
fn ptr_to_const_ref<'a, T>(ptr: *const T) -> &'a T {
    unsafe { &*ptr }
}

fn cstring_to_string(s: *const c_char) -> String {
    unsafe {
        assert!(!s.is_null());
        CStr::from_ptr(s).to_string_lossy().into_owned()
    }
}

fn string_to_cstring(string: &String, c_string: *mut c_char, c_string_len: usize) -> bool {
    let bytes = string.as_bytes();

    if c_string_len < bytes.len() + 1 {
        return false; // +1 for the null terminator
    }

    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), c_string as *mut u8, bytes.len());
        *c_string.add(bytes.len()) = 0; // Manually add null terminator
    }

    true
}

#[no_mangle]
pub extern "C" fn isox_term_type_to_string(term_type: c_uint) -> *const c_char {
    let c_str: &CStr = match TermType::from(term_type) {
        TermType::Int => CStr::from_bytes_with_nul(b"int\0").unwrap(),
        TermType::Float => CStr::from_bytes_with_nul(b"float\0").unwrap(),
        TermType::String => CStr::from_bytes_with_nul(b"string\0").unwrap(),
        TermType::Pid => CStr::from_bytes_with_nul(b"pid\0").unwrap(),
        TermType::Map => CStr::from_bytes_with_nul(b"map\0").unwrap(),
        TermType::List => CStr::from_bytes_with_nul(b"list\0").unwrap(),
        TermType::Binary => CStr::from_bytes_with_nul(b"binary\0").unwrap(),
        TermType::Atom => CStr::from_bytes_with_nul(b"atom\0").unwrap(),
        TermType::Tuple => CStr::from_bytes_with_nul(b"tuple\0").unwrap(),
        TermType::Resource => CStr::from_bytes_with_nul(b"resource\0").unwrap(),
        TermType::ResourceRef => CStr::from_bytes_with_nul(b"resourceRef\0").unwrap(),
        TermType::Bool => CStr::from_bytes_with_nul(b"bool\0").unwrap(),
    };
    c_str.as_ptr()
}

#[no_mangle]
pub extern "C" fn isox_priv_data(_env_ptr: *const Env) -> *mut c_void {
    unsafe { HOST_GLOBALS.priv_data() }
}

#[no_mangle]
pub extern "C" fn isox_set_priv_data(_env_ptr: *const Env, priv_data: *const c_void) {
    unsafe { HOST_GLOBALS.set_priv_data(priv_data) }
}

#[no_mangle]
pub extern "C" fn isox_create_resource_type(
    _env_ptr: *mut Env,
    destructor: ResourceDestructorFn,
    context: *const c_void,
) -> *const ResourceType {
    let resource_type = unsafe { HOST_GLOBALS.create_resource_type(destructor, context) };

    return Arc::into_raw(resource_type);
}

#[no_mangle]
pub extern "C" fn isox_alloc_env() -> *mut Env {
    let env = Arc::new(Env::new());
    Arc::into_raw(env) as *mut Env
}

#[no_mangle]
pub extern "C" fn isox_free_env(env_ptr: *const Env) {
    unsafe { Arc::from_raw(env_ptr) };
}

#[no_mangle]
pub extern "C" fn isox_copy_env(env_ptr: *mut Env) -> *mut Env {
    unsafe {
        Arc::increment_strong_count(env_ptr);
    }
    env_ptr
}

#[no_mangle]
pub extern "C" fn isox_copy_term(env_ptr: *mut Env, term_ptr: *const Term) -> *const Term {
    let env = ptr_to_mut_ref(env_ptr);
    let term_guard = ptr_to_arc(term_ptr);
    env.copy_term(term_guard);
    return term_ptr;
}

#[no_mangle]
pub extern "C" fn isox_is_equal(lhs: *const Term, rhs: *const Term) -> bool {
    unsafe {
        return (*lhs).value == (*rhs).value;
    }
}

#[no_mangle]
pub extern "C" fn isox_hash(term: *const Term) -> u64 {
    let mut hasher = DefaultHasher::new();
    unsafe {
        (*term).value.hash(&mut hasher);
    }
    return hasher.finish();
}

#[no_mangle]
pub extern "C" fn isox_make_resource(
    env_ptr: *mut Env,
    resource_type_ptr: *const ResourceType,
    resource_data: *const c_void,
) -> *const Term {
    let env = ptr_to_mut_ref(env_ptr);
    let resource_type = ptr_to_const_ref(resource_type_ptr);

    let term = Term {
        value: TermValue::Resource((
            resource_type.resource_type_id,
            RawPointer::new(resource_data),
        )),
    };

    env.add_term(term)
}

#[no_mangle]
pub extern "C" fn isox_read_resource(
    term: *const Term,
    resource_type: *mut *const ResourceType,
    resource_data_ptr: *mut *const c_void,
) -> bool {
    unsafe {
        match (*term).value {
            TermValue::Resource((resource_type_id, p)) => {
                let resource_record = HOST_GLOBALS.get_resource_record(resource_type_id);
                *resource_type = Arc::as_ptr(&resource_record.resource_type);
                *resource_data_ptr = p.as_ptr();
                true
            }
            _ => false,
        }
    }
}

#[no_mangle]
pub extern "C" fn isox_make_resource_ref(
    env_ptr: *mut Env,
    resource_ref: ResourceRefData,
) -> *const Term {
    let env = ptr_to_mut_ref(env_ptr);

    let term = Term {
        value: TermValue::ResourceRef((
            resource_ref.resource_type_id,
            RawPointer::new(resource_ref.obj),
        )),
    };

    env.add_term(term)
}

#[no_mangle]
pub extern "C" fn isox_make_int64(env_ptr: *mut Env, i: i64) -> *const Term {
    let env = ptr_to_mut_ref(env_ptr);

    let term = Term {
        value: TermValue::Int(i),
    };

    env.add_term(term)
}

#[no_mangle]
pub extern "C" fn isox_read_int64(term: *const Term, result: *mut i64) -> bool {
    unsafe {
        match (*term).value {
            TermValue::Int(x) => {
                *result = x;
                true
            }
            _ => false,
        }
    }
}

#[no_mangle]
pub extern "C" fn isox_make_bool(env_ptr: *mut Env, b: bool) -> *const Term {
    let env = ptr_to_mut_ref(env_ptr);
    let term = Term {
        value: TermValue::Bool(b),
    };

    env.add_term(term)
}

#[no_mangle]
pub extern "C" fn isox_read_bool(term: *const Term, result: *mut bool) -> bool {
    unsafe {
        match &(*term).value {
            TermValue::Bool(x) => {
                *result = *x;
                true
            }
            TermValue::Atom(x) => {
                if x == "true" {
                    *result = true;
                    true
                } else if x == "false" {
                    *result = false;
                    true
                } else {
                    false
                }
            }
            _ => false,
        }
    }
}

#[no_mangle]
pub extern "C" fn isox_make_float(env_ptr: *mut Env, f: f64) -> *const Term {
    let env = ptr_to_mut_ref(env_ptr);

    let term = Term {
        value: TermValue::Float(f),
    };

    env.add_term(term)
}

#[no_mangle]
pub extern "C" fn isox_read_float(term: *const Term, result: *mut f64) -> bool {
    unsafe {
        match (*term).value {
            TermValue::Float(x) => {
                *result = x;
                true
            }
            _ => false,
        }
    }
}

#[no_mangle]
pub extern "C" fn isox_make_string(env_ptr: *mut Env, s: *const c_char) -> *const Term {
    let env = ptr_to_mut_ref(env_ptr);
    let str = cstring_to_string(s);

    let term = Term {
        value: TermValue::String(str),
    };

    env.add_term(term)
}

#[no_mangle]
pub extern "C" fn isox_log(level: LogLevel, scope: *const c_char, message: *const c_char) -> bool {
    let mut ctx = IsoxTermSerializationContext::new();
    let log = FromExtensionMsg::Log {
        level,
        scope: cstring_to_string(scope),
        message: cstring_to_string(message),
    };
    log.isox_term_serialize(&mut ctx);
    unsafe { HOST_GLOBALS.send_message(ctx) }
}

#[no_mangle]
pub extern "C" fn isox_send_msg(
    _env_ptr: *mut Env,
    pid_ptr: *const Term,
    msg_ptr: *const Term,
) -> bool {
    let pid = ptr_to_arc(pid_ptr);
    let msg = ptr_to_arc(msg_ptr);
    let mut ctx = IsoxTermSerializationContext::new();
    let send_request = FromExtensionMsg::SendRequest {
        pid: pid.clone(),
        msg: msg.clone(),
    };
    send_request.isox_term_serialize(&mut ctx);
    unsafe { HOST_GLOBALS.send_message(ctx) }
}

#[no_mangle]
pub extern "C" fn isox_read_string_len(term: *const Term, len: *mut usize) -> bool {
    unsafe {
        match &(*term).value {
            TermValue::String(x) => {
                *len = x.len() + 1;
                true
            }
            _ => false,
        }
    }
}

#[no_mangle]
pub extern "C" fn isox_read_string(term: *const Term, s: *mut c_char, len: usize) -> bool {
    unsafe {
        match &(*term).value {
            TermValue::String(x) => string_to_cstring(x, s, len),
            _ => false,
        }
    }
}

#[no_mangle]
pub extern "C" fn isox_make_atom(env_ptr: *mut Env, s: *const c_char) -> *const Term {
    let env = ptr_to_mut_ref(env_ptr);
    let str = cstring_to_string(s);

    let term = Term {
        value: TermValue::Atom(str),
    };

    env.add_term(term)
}

#[no_mangle]
pub extern "C" fn isox_read_atom_len(term: *const Term, len: *mut usize) -> bool {
    unsafe {
        match &(*term).value {
            TermValue::Atom(x) => {
                *len = x.len() + 1;
                true
            }
            _ => false,
        }
    }
}

#[no_mangle]
pub extern "C" fn isox_read_atom(term: *const Term, s: *mut c_char, len: usize) -> bool {
    unsafe {
        match &(*term).value {
            TermValue::Atom(x) => {
                let c_str = CString::new(x.as_str()).expect("Failed to create CString");
                let bytes = c_str.as_bytes_with_nul();

                if len < bytes.len() {
                    false
                } else {
                    std::ptr::copy_nonoverlapping(bytes.as_ptr(), s as *mut u8, bytes.len());
                    true
                }
            }
            _ => false,
        }
    }
}

#[no_mangle]
pub extern "C" fn isox_make_list(
    env_ptr: *mut Env,
    num_terms: usize,
    terms: *const *const Term,
) -> *const Term {
    let env = ptr_to_mut_ref(env_ptr);
    let term_vec = ptrs_to_arcs(terms, num_terms);
    let term = Term {
        value: TermValue::List(term_vec),
    };

    env.add_term(term)
}

#[no_mangle]
pub extern "C" fn isox_read_list_len(term: *const Term, len: *mut usize) -> bool {
    unsafe {
        match &(*term).value {
            TermValue::List(x) => {
                *len = x.len();
                true
            }
            _ => false,
        }
    }
}

#[no_mangle]
pub extern "C" fn isox_read_list_item(term: *const Term, i: usize, item: *mut *const Term) -> bool {
    unsafe {
        match &(*term).value {
            TermValue::List(x) => {
                if i >= x.len() {
                    false
                } else {
                    *item = Arc::as_ptr(&x[i]);
                    true
                }
            }
            _ => false,
        }
    }
}

#[no_mangle]
pub extern "C" fn isox_make_tuple(
    env_ptr: *mut Env,
    num_terms: usize,
    terms: *const *const Term,
) -> *const Term {
    let env = ptr_to_mut_ref(env_ptr);
    let term_vec = ptrs_to_arcs(terms, num_terms);

    let term = Term {
        value: TermValue::Tuple(term_vec),
    };

    env.add_term(term)
}

#[no_mangle]
pub extern "C" fn isox_read_tuple_len(term: *const Term, len: *mut usize) -> bool {
    unsafe {
        match &(*term).value {
            TermValue::Tuple(x) => {
                *len = x.len();
                true
            }
            _ => false,
        }
    }
}

#[no_mangle]
pub extern "C" fn isox_read_tuple_item(
    term: *const Term,
    i: usize,
    item: *mut *const Term,
) -> bool {
    unsafe {
        match &(*term).value {
            TermValue::Tuple(x) => {
                if i >= x.len() {
                    false
                } else {
                    *item = Arc::as_ptr(&x[i]);
                    true
                }
            }
            _ => false,
        }
    }
}

#[no_mangle]
pub extern "C" fn isox_make_map(env_ptr: *mut Env) -> *const Term {
    let env = ptr_to_mut_ref(env_ptr);
    let term = Term {
        value: TermValue::Map(HashMap::new()),
    };

    env.add_term(term)
}

pub struct MapIterator<'a> {
    iter: Box<dyn Iterator<Item = (Arc<Term>, Arc<Term>)> + 'a>,
}

impl<'a> MapIterator<'a> {
    fn new(map: &'a HashMap<Arc<Term>, Arc<Term>>) -> MapIterator<'a> {
        MapIterator {
            iter: Box::new(map.iter().map(|(k, v)| (k.clone(), v.clone()))),
        }
    }
}

#[no_mangle]
pub extern "C" fn isox_get_map_iterator(term: *const Term, it: *mut *const MapIterator) -> bool {
    unsafe {
        match &(*term).value {
            TermValue::Map(x) => {
                let iterator = MapIterator::new(x);
                *it = Arc::into_raw(Arc::new(iterator));
                true
            }
            _ => false,
        }
    }
}

#[no_mangle]
pub extern "C" fn isox_read_map_iterator_next(
    it: *mut MapIterator,
    key: *mut *const Term,
    value: *mut *const Term,
) -> bool {
    unsafe {
        let iterator = &mut *it;
        match iterator.iter.next() {
            Some((it_key, it_value)) => {
                *key = Arc::as_ptr(&it_key);
                *value = Arc::as_ptr(&it_value);
                true
            }
            None => false,
        }
    }
}

#[no_mangle]
pub extern "C" fn isox_destroy_map_iterator(it: *mut MapIterator) {
    unsafe {
        let _ = Box::from_raw(it);
    }
}

#[no_mangle]
pub extern "C" fn isox_add_map_entry(
    env_ptr: *mut Env,
    key_ptr: *const Term,
    value_ptr: *const Term,
    map_ptr: *const Term,
) -> *const Term {
    let env = ptr_to_mut_ref(env_ptr);
    let key_guard = ptr_to_arc(key_ptr);
    let value_guard = ptr_to_arc(value_ptr);
    let map_guard = ptr_to_arc(map_ptr);

    // Ensure the map Term actually contains a Map variant
    if let TermValue::Map(ref existing_map) = map_guard.value {
        let mut new_map = existing_map.clone();
        new_map.insert(key_guard.clone(), value_guard.clone());
        let term = Term {
            value: TermValue::Map(new_map),
        };
        env.add_term(term)
    } else {
        panic!("add_map_entry called with non-map Term");
    }
}

#[no_mangle]
pub extern "C" fn isox_read_map_entry(
    term: *const Term,
    key: *const Term,
    result: *mut *const Term,
) -> bool {
    unsafe {
        match &(*term).value {
            TermValue::Map(x) => match x.get(&(*key)) {
                Some(value) => {
                    *result = Arc::as_ptr(value);
                    true
                }
                None => false,
            },
            _ => false,
        }
    }
}

#[no_mangle]
pub extern "C" fn isox_resource_to_resource_ref(term_ptr: *const Term) -> ResourceRefData {
    let term = unsafe { &*term_ptr };

    if let TermValue::Resource((i, p)) = term.value {
        ResourceRefData {
            resource_type_id: i,
            obj: p.as_ptr(),
        }
    } else {
        panic!("get_resource_term_ref called with non-resource Term");
    }
}

#[no_mangle]
pub extern "C" fn isox_make_binary(
    env_ptr: *mut Env,
    size: usize,
    out_term: *mut *const Term,
) -> *mut c_void {
    let env = ptr_to_mut_ref(env_ptr);

    let shm_segment = ShmSegment::new(size);
    let addr = shm_segment.get_addr();
    let term = Term {
        value: TermValue::Shm(shm_segment),
    };

    unsafe {
        *out_term = env.add_term(term);
    };

    addr as *mut c_void
}

#[no_mangle]
pub extern "C" fn isox_read_binary(
    term: *const Term,
    result: *mut *const c_void,
    size: *mut usize,
) -> bool {
    unsafe {
        match &(*term).value {
            TermValue::Shm(shm_segment) => {
                *result = shm_segment.get_addr();
                *size = shm_segment.size;
                true
            }
            _ => false,
        }
    }
}

#[no_mangle]
pub extern "C" fn isox_make_sub_binary(
    env_ptr: *mut Env,
    binary_ptr: *const Term,
    start: usize,
    length: usize,
) -> *const Term {
    let env = ptr_to_mut_ref(env_ptr);
    let binary = ptr_to_arc(binary_ptr);

    match binary.borrow() {
        Term {
            value: TermValue::Shm(_),
        } => {
            let term = Term {
                value: TermValue::SubBinary(binary, start, length),
            };

            env.add_term(term)
        }
        _ => {
            panic!("make_sub_binary called with non-binary");
        }
    }
}

#[repr(C)]
#[derive(Debug)]
pub struct IsoxFunctionTable {
    pub name: *const c_char,
    pub load: extern "C" fn(
        context: *const c_void,
        env: *mut Env,
        arguments: *const Term,
        priv_data: *mut *const c_void,
    ) -> *const Term,
    pub query:
        extern "C" fn(context: *const c_void, env: *mut Env, arguments: *const Term) -> *const Term,
    pub create_instance:
        extern "C" fn(context: *const c_void, env: *mut Env, arguments: *const Term) -> *const Term,
    pub query_instance: extern "C" fn(
        context: *const c_void,
        env: *mut Env,
        instance: *const Term,
        arguments: *const Term,
    ) -> *const Term,
    pub update_instance: extern "C" fn(
        context: *const c_void,
        env: *mut Env,
        instance: *const Term,
        arguments: *const Term,
    ) -> *const Term,
    pub destroy_instance:
        extern "C" fn(context: *const c_void, env: *mut Env, resource: *const Term) -> *const Term,
    pub unload: extern "C" fn(context: *const c_void),
    pub context: *const c_void,
}

#[no_mangle]
pub extern "C" fn do_not_use(_table: &IsoxFunctionTable) {}

pub mod logger {
    use lazy_static::lazy_static;
    use std::fs::OpenOptions;
    use std::sync::Mutex;
    use std::{env, process};

    lazy_static! {
        pub static ref LOG_FILE: Mutex<Option<std::fs::File>> = {
            let tmpdir = env::var("TMPDIR").unwrap_or_else(|_| "/tmp".to_string());
            let pid = process::id(); // Get the current process ID
            let filename = format!("{}/host-debug-{}.log", tmpdir, pid);
            let file = OpenOptions::new()
                .append(true)
                .create(true)
                .open(filename)
                .ok();
            Mutex::new(file)
        };
    }

    #[macro_export]
    macro_rules! log_to_file {
        ($($arg:tt)*) => {
            {
                use std::io::Write;
                let log_file = $crate::logger::LOG_FILE.lock().unwrap();
                if let Some(file) = log_file.as_ref() {
                    let mut file = file;
                    writeln!(file, $($arg)*).ok();
                }
            }
        };
    }
}
