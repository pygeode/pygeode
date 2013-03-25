#TODO: change Last-Modified header value to the time of the server instantiation,
#   not the time of the client request??

# OPeNDAP client / server code
# NOTE: only a subset of the OPeNDAP interface is implemented.  Use at your own risk.

# conversion from numpy dtype to opendap type
# note: opendap doesn't support 64-bit integers or complex numbers - cast as Float64
np2dap = {'uint8':'Byte', 'int8':'Int16', 'uint16':'UInt16', 'int16':'Int16',
          'uint32':'UInt32', 'int32':'Int32', 'uint64':'Float64', 'int64':'Float64',
          'float32':'Float32', 'float64':'Float64', 'string8':'String'}

dap2np = {'byte':'uint8', 'uint16':'uint16', 'int16':'int16',
          'uint32':'uint32', 'int32':'int32', 'float32':'float32', 'float64':'float64',
          'string':'string8'}

supported_type = dict((k,dap2np[v.lower()]) for k,v in np2dap.iteritems())

# Supporting library
from pygeode.formats import opendapcore as lib


###############################################################################
# ENcoding data to OpenDAP
###############################################################################



# trap data I/O errors gracefully
def get_data_trap_io (view, var):
  from warnings import warn
  import numpy as np
  try:
    indata = view.get(var)
#  except Exception as e:
#  except Exception:
  except Exception, e:
#    warn ("Received an exception on %s -- filling with NaN and continuing anyway"%str(var))
    warn ("Received the following Exception:\n%s\n -- filling with NaN and continuing anyway"%e.args[0])
    indata = np.empty (view.shape, var.dtype)
    indata[()] = float('NaN')
  return indata

# Write variable data to an http stream
# Since this is done on the server side, where the user has no direct control,
# it is more convenient to trap errors and at least send *something* (i.e. NaN)
# to the client, so their program doesn't crash due to something beyond their control.
def write_xdr(var, wfile):
  import struct
  import numpy as np
  from pygeode.view import View

  lenstr = struct.pack('!2l', var.size, var.size)
  wfile.write(lenstr)

  # Break the values into memory-friendly chunks
  if hasattr (var, 'values'):
    values_iter = [var.values]
  else:
    view = View(var.axes)
    # Trap and handle any I/O errors
    viewloop = view.loop_mem()
    #TODO: make this more general - should we be futzing around with the axes at this level
    # Break it up even further along the time axis?  (so we don't start a long process through the whole dataset)
    if var.naxes > 2:
      new_viewloop = []
      for v in viewloop:
        for s in v.integer_indices[0]:
          new_viewloop.append(v.modify_slice(0,[s]))
      viewloop = new_viewloop

    values_iter = (get_data_trap_io(v,var) for v in viewloop)

  for values in values_iter:

    daptype = np2dap[values.dtype.name]
    if daptype in ('Byte','String'):
#      # Do byte encoding here
#      raise Exception
      values = np.ascontiguousarray(values, 'uint8');
      s = lib.int8toStr(values)
    elif daptype in ('UInt16', 'Int16', 'UInt32', 'Int32'):
      values = np.ascontiguousarray(values, 'int32')
      s = lib.int32toStr(values)
    elif daptype == 'Float32':
      values = np.ascontiguousarray(values, 'float32')
      s = lib.float32toStr(values)
    elif daptype == 'Float64':
      values = np.ascontiguousarray(values, 'float64')
      s = lib.float64toStr(values)

    wfile.write(s)

  # end of write_xdr


#####################################


# An automatically chunked HTTP stream
# (since BaseHTTPServer doesn't handle it for us)
class Chunked:
  def __init__ (self, wfile):
    self._wfile = wfile
  def write (self, data):
    self._wfile.write("%x\r\n"%len(data))
    self._wfile.write(data)
    self._wfile.write("\r\n")

# An HTTP handler for OpenDAP requests
# (does nothing until it's added to an http server)
# i.e., s = pygeode.server.web.HTTPDaemon(8080, DAPHandler(dataset))
class DAPHandler:

  def __init__(self, dataset):
    from pygeode.formats.cfmeta import encode_cf
    self.dataset = dataset = encode_cf(dataset)

  def err (self, msg, h):
    h.send_response (404)
    h.send_header('Transfer-Encoding', 'chunked')
    h.end_headers()
#    self.send_data(h, '<h1>How is babby formed?</h1><br><br>')
    self.send_data(h, msg)
    self.send_data(h, '')

  # Send data and close the connection
  def send_data (self, h, data, binary=False):
    h.wfile.write("%x\r\n"%len(data))
    h.wfile.write(data)
    h.wfile.write("\r\n")
#    h.wfile.flush()

  def handle (self, h, relpath, headeronly=False):
    from datetime import datetime
    import re
    import urllib
    import numpy as np
    from warnings import warn
    from cStringIO import StringIO

    datestr = datetime.utcnow().strftime("%a, %d %b %Y %H:%M:%S GMT")

    extensions = '.das', '.dds', '.dods'

    path = relpath
    if '?' in path: path, args = path.rsplit('?',1)
    else: args = ''

    for ext in extensions:
      if path.endswith(ext):
        path, request_type = path[:-len(ext)], ext.lstrip('.')
        break
    else:
      request_type = ''

    # The name to associate with this data (exclude the leading /)
    filename = path[1:]

    # Was the URL entered from a web browser (i.e., no extension?)
    # Print some helpful pointers, but don't start processing the data!
    if request_type == '':
      h.send_response (200, "OK")
      h.send_header ("content-type", "text/html; charset=UTF-8")
      h.send_header ("connection", "close")
      h.end_headers()
      h.wfile.write ("Sorry, but the data can't be viewed directly in a web browser.<br>\
                      You'll have to paste the URL into an OPeNDAP-enabled client, such as \
                      <a href='http://www.epic.noaa.gov/java/ncBrowse/'>ncBrowse</a> \
                      or <a href='http://www.unidata.ucar.edu/software/idv/'>IDV</a>.")
      return

#    # read the headers (as dictionary)
#    headers = self.headers.dict

    send_header = h.send_header

    if request_type not in ('dds', 'das', 'dods'):
      return self.err("how is babby formed?", h)

    if request_type == 'dods': request_type = 'data'


    def check (dataset, varname, axisname):
      if varname not in dataset:
        raise Exception("Unknown variable '%s'"%varname)
      var = dataset[varname]
      if axisname is not None and not var.hasaxis(axisname):
        raise Exception("'%s' has no member '%s'"%(varname,axisname))

    # Add a var stub
    def var_stub (fields, dataset, varname):
      if varname not in fields: fields[varname] = {}
      return fields[varname]

    # Add an axis
    def add_axis(fields, dataset, varname, axisname, sl):
      check (dataset, varname, axisname)

      if len(sl) == 0: sl = [slice(None)]
      if len(sl) != 1:
        raise Exception("Multidimensional slices used for axis '%s.%s'"%(varname,axisname))

      x = var_stub (fields, dataset, varname)
      if axisname in x and x[axisname] != sl:
        raise Exception("Inconsistent slicing for '%s.%s'"%(varname,axisname))
      x[axisname] = sl


    # Add a var
    def add_var (fields, dataset, varname, slices, recurse=False):
      check (dataset, varname, None)
      var = dataset[varname]
      # Empty slices -> no slicing
      if len(slices) == 0: slices = [slice(None)]*var.naxes
      if len(slices) != var.naxes:
        raise Exception("Incorrect number of slices for '%s'"%varname)
      x = var_stub (fields, dataset, varname)
      if varname in x and x[varname] != slices:
        raise Exception("Inconsistent slicing for '%s'"%varname)
      x[varname] = slices
      # Add the variable's axes? (if it's not itself an axis)
      if recurse is True and varname in dataset.vardict:
        for axis,sl in zip(var.axes,slices):
          add_axis (fields, dataset, varname, axis.name, [sl])

    # Return string representation of a var/axis
    # (also, add the var to the list of arrays being returned)
    # Note: 'axisname' could be the name of the var -
    # this is due to the way opendap structures the data.
    # There's an outer container for the var (Grid/Structure), and then the
    # inner data array and axes.
    def make_entry (fields, dataset, varname, axisname, arrays):

      # Slice the thing
      sl = fields[varname][axisname]
      if axisname == varname:
        var = dataset[varname]
      else:
        var = dataset[varname].getaxis(axisname)
      var = var._getitem_asvar(sl)
      arrays.append(var)

      dtype = var.dtype.name
      if dtype not in np2dap:
        raise Exception("can't convert %s to an opendap type" % dtype)
      daptype = np2dap[dtype]

      s = daptype + " " + var.name
      for axis in var.axes:
        # Perform slicing on the axis??
        s += "[%s = %s]"%(axis.name,len(axis))
      return s


    # Okay, now figure out what to do with the request - what to return
    # (build up a dictionary structure of what was selected).
    # fields is a dataset dictionary (referencing vars and axes).
    # Each entry is itself a dictionary, containing the axes and the data.
    # Each of those entries is in turn a slice going into the data.
    dataset = self.dataset
    fields = {}
    # Buffer for text output
    buf = StringIO()
    # Keep track of which arrays we describe (in case we need the data as well)
    arrays = []

    # If no arguments given, then return *everything*
    if len(args) == 0:
      for varname in dataset.axisdict.keys() + dataset.vardict.keys():
        add_var (fields, dataset, varname, slices=[], recurse=True)

    # Otherwise, have to parse through the arguments, find out what we need.
    else:
      for varname in args.split(','):
        varname = urllib.unquote(varname)  # unencode the characters
        # Extract the slicing from the variable name
        if '[' in varname:
          ind = varname.index('[')
          varname, slices = varname[:ind], varname[ind:]
        else: slices = ''
        slices = re.findall('\[[0-9:]+\]', slices)
        slices = [sl[1:-1] for sl in slices]  # strip out [ and ]
        for i,sl in enumerate(slices):
          try:
            sl = map(int, sl.split(':'))
          except ValueError:
            return self.err("bad slicing syntax", h)
          if len(sl) == 1:  slices[i] = slice(sl[0], sl[0]+1)
          elif len(sl) == 2: slices[i] = slice(sl[0], sl[1]+1)
          elif len(sl) == 3: slices[i] = slice(sl[0], sl[2]+1, sl[1])
          else: return self.err("incorrect number of slice parameters", h)

        if varname.count('.') > 1: return self.err("invalid hierarchy %s"%varname, h)

        try:
          # Axis / inner var structure requested?
          if varname.count('.') == 1:
            varname, axisname = varname.split('.')
            if varname == axisname:
              add_var (fields, dataset, varname, slices, recurse=False)
            else:
              add_axis (fields, dataset, varname, axisname, slices)
          # Full var requested?
          else:
            add_var (fields, dataset, varname, slices, recurse=True)
        except Exception, e:
          return self.err("Error: "+e.args[0], h)

#    return self.err("okay...\n%s"%fields, h)


    # Attributes? (all attributes, no subsetting is respected here)
    if request_type == 'das':
      buf.write("Attributes {\n")
      for var in dataset.axes + dataset.vars + [dataset]:
        varname = 'NC_GLOBAL' if var is dataset else var.name
        buf.write("    %s {\n"%varname)
        for name,value in var.atts.iteritems():
#          buf.write("        ?? %s = %s\n"%(k,v))
          ##
          # determine the type of the attribute
          if isinstance(value,str):
            # Escape out anything which would mess up the string representation??
            for c1, c2 in [('\\', r'\\'), ("\n", r"\n"), ('"',r'\"'), ("'", r"\'"), ("\t",r"\t")]:
              value = value.replace(c1,c2)
            buf.write('        String %s "%s";\n' % (name, value))
          # try to wrap as a numpy array
          else:
            value = np.array(value)
            dtype = value.dtype.name
            if dtype not in np2dap:
              warn ("can't convert %s to an opendap type" % dtype, stacklevel=4)
              continue
            daptype = np2dap[dtype]
            # Scalar?
            if value.ndim == 0:
              buf.write("        %s %s %s;" % (daptype, name, value))
            elif value.ndim == 1:
              buf.write("%s %s %s;" % (daptype, name, ', '.join(str(v) for v in list(value))))
            else:
              warn ("attribute %s has too many dimensions" % (name), stacklevel=4)
              continue
          ##

        buf.write("    }\n")
      buf.write("}\n")

    # Header for dds/data
    else:
      buf.write("Dataset {\n")
      # Use the order from the dataset
      # Axes
      for axis in dataset.axes:
        if axis.name in fields:          
          buf.write("    %s;\n"%make_entry(fields,dataset,axis.name,axis.name,arrays))
      for var in dataset.vars:
        if var.name not in fields: continue
        x = fields[var.name]
        # Special case: all axes are requested - can form a Grid
        # (check that the axes are consistent with the var slicing)
        full_map = True
        if var.name not in x: full_map = False
        elif not all(axis.name in x for axis in var.axes): full_map = False
        elif x[var.name] != [x[axis.name][0] for axis in var.axes]: full_map = False
        if full_map is True:
          buf.write("    Grid {\n");
          buf.write("      Array:\n");
          buf.write("        %s;\n"%make_entry(fields,dataset,var.name,var.name,arrays))
          buf.write("      Maps:\n");
          for axis in var.axes:
            buf.write("        %s;\n"%make_entry(fields,dataset,var.name,axis.name,arrays))
          buf.write("    } %s;\n"%var.name)
        # Otherwise, return a more generic Structure
        else:
          buf.write("    Structure {\n");
          if var.name in x:
            buf.write("        %s;\n"%make_entry(fields,dataset,var.name,var.name,arrays))
          for axis in var.axes:
            if axis.name not in x: continue
            buf.write("        %s;\n"%make_entry(fields,dataset,var.name,axis.name,arrays))
          buf.write("    } %s;\n"%var.name)
      buf.write("} %s;\n"%filename)



    # Send the header

    h.send_response (200, "OK")

    send_header('Last-Modified', datestr)
    send_header('Content-Description', 'dods_'+request_type)
    # include these, even though we don't implement a full opendap server
    send_header('XDODS-Server', 'dods/3.2')
    send_header('XOPeNDAP-Server', 'bes/3.7.2, libdap/3.9.3, dap-server/ascii/3.9.3, freeform_handler/3.7.12, fileout_netcdf/0.9.3, hdf4_handler/3.7.14, hdf5_handler/1.3.3, netcdf_handler/3.8.3, dap-server/usage/3.9.3, dap-server/www/3.9.3')
    send_header('XDAP', '3.2')
#    send_header('Connection', 'close')

    # We will be doing chunked encoding
    send_header('Transfer-Encoding', 'chunked')
    wfile = Chunked(h.wfile)

    if request_type in ('dds','das'):
      send_header('Content-Type', 'text/plain; charset=UTF-8')
    else:
      send_header('Content-Type', 'application/octet-stream')

    h.end_headers()

    if headeronly: return

    # Send the text portion
    wfile.write(buf.getvalue())

    # Send the arrays?
    if request_type == 'data':
      wfile.write("Data:\n")
      for var in arrays:
        write_xdr(var, wfile)

    # Try sending one last, empty chunk to indicate the end of this transfer.
    # Don't panic if this fails... we already sent the data.
    try:
      self.send_data(h, '')
    except Exception: pass

    return


from pygeode.server.web import Dir
class DAP_Dir (Dir):
  def handle (self, h, relpath, headeronly=False):
#    print "dap_dir handler:", relpath

    path = h.path
    if not path.endswith('/'): path += '/'

    # Directory listing?
    if relpath == "": relpath = "/"
    if relpath == "/":
      h.send_response(200)
      h.send_header ("Content-Type", "text/html; charset=UTF-8")
      h.send_header ("Connection", "close")
      h.end_headers()

      if headeronly: return

      self.dir_header (h, relpath)
      h.wfile.write ("<table border=0 cellpadding=5>\n")
      # Sort the file names
      names = sorted(self.nodes.iterkeys())
      for name in names:
        node = self.nodes[name]
        # directory?
        if isinstance (node,Dir):
          h.wfile.write("<tr><td><a href='%s%s/'>%s/</a></td><td>---</td><td>---</td></tr>\n"%(path,name,name))
          continue
        if not isinstance (node, DAPHandler):
          h.wfile.write("<tr><td colspan=3><a href='%s%s'>%s</a></td></tr>\n"%(path,name,name))
          continue
        # otherwise, assume we have an opendap node
        h.wfile.write ("<tr><td><a href='%s%s'>%s</a> &nbsp; &nbsp; &nbsp; </td><td><a href='%s%s.dds'>dds</a></td><td><a href='%s%s.das'>das</a></td></tr>\n"%(path,name,name,path,name,path,name))
      h.wfile.write ("</table>\n")
      self.dir_footer (h, relpath)
    # Otherwise, a DAP object
    # Don't bother adjusting the relative path - it's ignored by dap objects
    else:
      assert len(relpath) > 0 and relpath[0] == "/"
      # Map to the first matching one?
      for name,node in self.nodes.iteritems():
        if relpath.startswith("/"+name+".dds"): return node.handle (h, relpath, headeronly=headeronly)
        if relpath.startswith("/"+name+".das"): return node.handle (h, relpath, headeronly=headeronly)
        if relpath.startswith("/"+name+".dods"): return node.handle (h, relpath, headeronly=headeronly)
        if relpath == "/"+name and not isinstance(node, Dir):
#          print "relpath '%s' matches node named '%s'"%(relpath,name)
          return node.handle (h, relpath, headeronly=headeronly)  # The user tried to load the data in a web browser?
      # Otherwise, try it as a directory name or other object??  (fall back to Dir behaviour?)
#      print "deferring to Dir.handle"
      Dir.handle(self, h, relpath, headeronly=headeronly)


# Hold all existing OPeNDAP servers being run from this instance
class _SERVERS:
  def __init__(self):
    self.serverdict={}
    self.threaddict={}
SERVERS=_SERVERS()

# Shortcut for hosting data through the OPeNDAP interface
def serve (path, dataset, port=8080):
  from pygeode.server.web import MyServer_threaded2
  import threading
  from pygeode.dataset import asdataset
  # Remove extra /'s
  while '//' in path: path = path.replace('//','/')
  # Remove any leading and trailing /
  if path.startswith('/'): path = path[1:]
  if path.endswith('/'): path = path[:-1]
  # Break up into directories and 'file' name
  parts = path.split('/')
  dirnames = parts[:-1]
  fname = parts[-1]
  # Check if we have a server available already
  if port not in SERVERS.serverdict:
    # Make a new server, with an empty root directory
    root = DAP_Dir()
    server = MyServer_threaded2(port, root)
    threading.Thread(target=server.serve_forever).start()
    print "Started an OPeNDAP server listening on port %s"%port
    SERVERS.serverdict[port] = server
  else:
    server = SERVERS.serverdict[port]
  # Get the working directory
  cwd = server.root_handler
  # Make sure the full path is available
  for dname in dirnames:
    if dname not in cwd.nodes:
      cwd.nodes[dname] = DAP_Dir()
    cwd = cwd.nodes[dname]
    assert hasattr(cwd,'nodes'), "'%s' is already defined as something other than a directory?"%dname

  # Share the file
  assert fname not in cwd.nodes, "'%s' is already being served"%path
  cwd.nodes[fname] = DAPHandler(asdataset(dataset))

# Kill an opendap server
def kill (port=8080):
  assert port in SERVERS.serverdict, "no server running on port %s?"%port
  server = SERVERS.serverdict[port]
  server.server_close()
  print "Killed OPeNDAP server on port", port
  del SERVERS.serverdict[port]


###############################################################################
# DEcoding data from OpenDAP
###############################################################################


# Split a string up into tokens
class tokenize:
#  import psyco
  def __init__(self, s):
    import re
    # First, break down the string into quoted / unquoted 
    string1 = r'(?<=\s)"(?:\\"|[^"])*[^\\]"'
    string2 = r"(?<=\s)'(?:\\'|[^'])*[^\\]'"
    unstring = '[^"\']*'
    tokens = re.findall('('+string1+'|'+string2+'|'+unstring+')', s)
#    print '??', tokens
    # Break down tokens even further if not surrounded by quotes
#    pattern = "\s*([a-zA-Z0-9_.+-]+|;|:|{|}|\[|\]|=|\"[^\"]*\"|'[^']*')"
    pattern = r"\s*([a-zA-Z0-9_.+-]+|;|:|{|}|\[|\]|=)"
    # remove empty tokens
    assert len(tokens[-1]) == 0
    tokens = tokens[:-1]
    tokens = [ t for tok in tokens for t in (re.findall(pattern,tok) if tok[0] not in ('"',"'") else [tok]) ]
#    self.tokens = re.findall(pattern, s, re.DOTALL)
    self.tokens = tokens
    self.i = 0
  def __iter__(self): return self
  def peek(self):
    if self.i == len(self.tokens): raise StopIteration
    return self.tokens[self.i]
#  psyco.bind(peek)
  def next(self):
    t = self.peek()
    self.i += 1
    return t
#  psyco.bind(next)
  def expect (self, e):
    t = self.next()
    if t.lower() != e.lower():
      raise Exception ("expected '%s', found '%s'"%(e,t))
#  psyco.bind(expect)

def parse_array (s):
  # Return a list of tuples, of the form (daptype, name, dimnames, shape)

  daptype = s.next().lower()
  assert daptype in dap2np.keys(), "unknown type '%s'"%daptype

  name = s.next()

  dimnames = []
  shape = []
  
  while True:
    t = s.next()
    if t == ";": return (daptype, name, dimnames, shape)

    assert t == "[", "unknown syntax"
    size = s.next()
    try:
      shape.append(int(size))
      dimnames.append(None)
    except ValueError:
      dimname = size
      dimnames.append(dimname)
      s.expect("=")
      size = int(s.next())
      shape.append(size)
    s.expect ("]")

def parse_grid (s):
  s.expect ("grid")
  s.expect ("{")
  s.expect ("array")
  s.expect (":")
  arr = parse_array(s)
  ndims = len(arr[3])
  s.expect ("maps")
  s.expect (":")
  # Match to maps
  for name,size in zip(arr[2], arr[3]):
    assert name is not None, "grids must have named dimensions"
    m = parse_array(s)
    assert len(m[2]) == len(m[3]) == 1, "axes cannot have more than one dimension"
    mname = m[2][0]
    msize = m[3][0]
    assert mname == name, "dimension name does not match: %s <-> %s"%(mname,name)
    assert msize == size, "dimension size does not match"

  s.expect ("}")
  name = s.next()
  assert name == arr[1], "%s != %s"%(name,arr[1])
  s.expect (";")
  return arr

# Parse through a dataset, return the result (no values loaded yet)
def parse_dataset (s):

  # Must have a dataset
  s.expect("dataset")
  s.expect("{")

  out = []

  try:
   while True:

    t = s.peek()

    if t == '}':
      s.next()
      return out

    #atomic/array?
    if t.lower() in dap2np:
      out.append (parse_array(s))
    elif t.lower() == 'grid':
      out.append (parse_grid(s))
    else:
      raise Exception ("unhandled type "+t)
  except StopIteration:
   pass

def parse_attributes (s):
  import re
  s.expect("attributes")
  s.expect("{")
  atts = []
  # Loop over all variables
  while s.peek() != "}":
    varname = s.next()
    varatts = []
    atts.append([varname, varatts])
    s.expect("{")
    # Loop over all attributes
    while s.peek() != "}":
      daptype = s.next().lower()
      attname = s.next()
      attvalue = []
      # Load array?
      while s.peek() != ";":
        x = s.next()
        # cast into the proper type
        if daptype == "string":
          pass
          x = re.sub(r'(?<=[^\\])\\n', '\n', x) # decode newlines
          x = re.sub(r'(?<=[^\\])\\t', '\t', x) # decode tabs
          x = re.sub(r'(?<=[^\\])\\"', '"', x) # decode quotes
          x = re.sub(r"(?<=[^\\])\\'", "'", x) # decode quotes
          x = re.sub(r'\\\\', r'\\', x) # decode backslashes
          x = x[1:-1]  # strip off enclosing quotes
        elif daptype in ("int16", "uint16", "int32", "uint32"): x = int(x)
        elif daptype in ("float32", "float64"): x = float(x)
        else: raise Exception
        attvalue.append(x)
      # Unwrap scalars
      assert len(attvalue) > 0, "no attribute values found!"
      if len(attvalue) == 1: attvalue = attvalue[0]
      varatts.append([daptype, attname, attvalue])
      s.expect(";")
    s.expect("}")
  s.expect("}")
  return atts

def print_dataset(d):
  for (daptype,name,dimnames,shape) in d:
    print daptype, name,
    for dimname,size in zip(dimnames,shape):
      if dimname is not None:
        print "[%s = %i]"%(dimname,size),
      else:
        print "[%i]"%size,
    print

from pygeode.var import Var
class OpenDAP_Var(Var):
  def __init__(self, name, axes, dtype, url):
    self.name = name
    self.dapname = name  # just in case somebody futzes with the name
    self.url = url
    Var.__init__(self, axes, dtype)
  #TODO: allow strides (since OpenDAP handles strides natively)
  def getvalues (self, start, count):
    subset = ''.join("[%s:%s]"%(s,s+c-1) for s,c in zip(start,count))
    return  load_array (self.url+".dods?"+self.dapname+subset)

#TODO: allow multiple arrays to be loaded at once
def load_array (url):
  import struct
  import numpy as np

  # Read binary data (with ascii header)
  data = readurl(url)
  # Strip off ascii header
  lookfor = "Data:\n"
  try:
    i = data.index(lookfor)
  except ValueError:
    print "unrecognized return values.  raw dump:"
    print data
    raise
  xdr = data[i+len(lookfor):]
  axis_dataset = parse_dataset(tokenize(data[:i]))
  # Should only have the 1 axis
  assert len(axis_dataset) == 1, "?? %s"%axis_dataset
  axis_dataset = axis_dataset[0]
  # Get the length and type of the axis
  daptype, name, dimnames, shape = axis_dataset


  # Load the data
  size = int(np.product(shape))
  assert size > 0

  length, length2 = struct.unpack('!2l', xdr[:8])
  assert length == length2, "bad xdr length: %i != %i"%(length,length2)
  assert length == size, "expected size %i, found %i"%(size,length)


  # Strip off length
  xdr = xdr[8:]
  if daptype == 'byte':
    # Do byte encoding here
    raise Exception
  elif daptype in ('uint16', 'int16', 'uint32', 'int32'):
#    arr = struct.unpack('!%sl'%size, xdr[:4*size])
    arr = lib.str2int32(xdr)
  elif daptype == 'float32':
#    arr = struct.unpack('!%sf'%size, xdr[:4*size])
    arr = lib.str2float32(xdr)
  elif daptype == 'float64':
#    arr = struct.unpack('!%sd'%size, xdr[:8*size])
    arr = lib.str2float64(xdr)
  else: raise Exception

  return arr[:size].reshape(shape)


def readurl (url, hosts={}):
  from httplib import HTTPConnection

  i = url.find('http://')
  assert i == 0, "I don't know how to handle this protocol"
  url = url[7:]
  host, path = url.split('/', 1)
  path = '/' + path

  # Check if we already have a connection
  if host not in hosts:
    conn = hosts[host] = HTTPConnection(host)
#    conn.connect()
  else:
    conn = hosts[host]

  conn.connect()

  conn.request("GET", path)
  resp = conn.getresponse()
  assert resp.status == 200, str(resp.status)+" "+resp.reason
  data = resp.read()
  return data

# Load an opendap dataset
def open (url):
  from pygeode.axis import NamedAxis
  from pygeode.formats.cfmeta import decode_cf
  from pygeode.dataset import asdataset
  from warnings import warn

  warn ("opendap.open is deprecated.  Please use netcdf.open, if you have a netcdf library with opendap support compiled in.  It is more flexible, faster, and standard.", stacklevel=2)

  dds = readurl(url+".dds")

  # Get list of variables, axes, and types from the url
  dataset = parse_dataset(tokenize(dds))

  # Get metadata
  das = readurl(url+".das")
  attributes = parse_attributes(tokenize(das))

  # Get the axes
  axis_names = []
  axis_shapes = []
  for (daptype,name,dimnames,shape) in dataset:
    for dimname,length in zip(dimnames,shape):
      #TODO: handle unnamed axes
      assert dimname is not None, "can't handle unnamed axes"
      if dimname not in axis_names:
        axis_names.append(dimname)
        axis_shapes.append(int(length))
  # Construct the axes
  dataset_axes = []
  for axis, length in zip(axis_names, axis_shapes):
    try:
      values = load_array(url+".dods?"+axis)
    # catch any problematic axes
    # i.e., nbnds in http://test.opendap.org/dap/data/nc/sst.mnmean.nc.gz
    # (which doesn't have any values!!)
    except ValueError:
      from warnings import warn
      from numpy import arange
      warn ("No values found for axis '%s'"%axis, stacklevel=2)
      values = arange(length)
    atts = {}
    # get the axis attributes
    for A in attributes:
      # Match to axis name
      if A[0] == axis:
       atts = dict([(k,v) for t,k,v in A[1]])

    # Construct this axis
    dataset_axes.append(NamedAxis(values=values, name=axis, atts=atts))

  # Construct the vars
  dataset_vars = []
  for daptype, name, dimnames, shape in dataset:
    # Don't treat axes as vars
    if name in axis_names: continue
    axes = [a for n in dimnames for a in dataset_axes if a.name == n]
    dtype = dap2np[daptype]
    var = OpenDAP_Var(name, axes, dtype, url)
    atts = {}
    # get the var attributes
    for A in attributes:
      # Match to var name
      if A[0] == name:
       atts = dict([(k,v) for t,k,v in A[1]])
    if len(atts) > 0: var.atts = atts

    dataset_vars.append(var)

  # Create a dataset
  dataset =  decode_cf(dataset_vars)

  # Global attributes?
  for A in attributes:
    if A[0] == 'NC_GLOBAL': dataset.atts = dict([(k,v) for t,k,v in A[1]])

  return dataset
