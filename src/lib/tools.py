import datetime
import json
import io
import re
import os
import csv
import yaml
from yaml.representer import SafeRepresenter
from ruamel.yaml import YAML







# ================================================================================
# H E L P E R   F U N C T I O N S


def strip_lines(s):
    # For each line in s, strip the line and return the result.
    return '\n'.join([line.strip() for line in s.split('\n')])
    

def to_seconds(s):
    if isinstance(s, float):
        return s
    if isinstance(s, int):
        return s
    if isinstance(s, str):
        s = s.strip()
        if is_float(s):
            return float(s)
        seconds = 0
        while True:
            # For all regex matches of an integer followed by a string of characters, return the integer.
            re = getRegex(r"(?P<N>\d+(\.\d+)?)(?P<U>[^\d]+)")
            match = re.match(s)
            if match:
                n = float(match.group("N"))
                u = match.group("U").lower()
                if u in ['s', 'sec', 'secs', 'second', 'seconds']:
                    seconds += n
                elif u in ['m', 'min', 'mins', 'minute', 'minutes']:
                    seconds += 60 * n
                elif u in ['h', 'hr', 'hrs', 'hour', 'hours']:
                    seconds += 3600 * n
                elif u in ['d', 'day', 'days']:
                    seconds += 86400 * n
                elif u in ['w', 'week', 'weeks']:
                    seconds += 604800 * n
                elif u in ['mo', 'month', 'months']:
                    seconds += 2629746 * n
                elif u in ['y', 'year', 'years']:
                    seconds += 86400 * n
                s = s.replace(match.group(0), '', 1)
            else:
                break
        return seconds
    return 0


def is_low_memory(threshold_percent=85):
    import psutil
    # Get memory statistics
    mem = psutil.virtual_memory()
    
    # 'available' is the best metric—it accounts for cached memory 
    # that the OS can give back to processes immediately.
    available_gb = mem.available / (1024**3)
    
    print(f"Memory Usage: {mem.percent}%")
    print(f"Available RAM: {available_gb:.2f} GB")

    if mem.percent > threshold_percent:
        print("⚠️ WARNING: Low RAM detected! Consider closing background apps.")
        return True
    return False

# Run the check
is_low = is_low_memory()


def from_metric(s):
    """Convert a string like '123K' to an integer."""
    s = str(s).strip()
    if s.endswith('K'):
        return int(s[:-1]) * 1024
    elif s.endswith('M'):
        return int(s[:-1]) * 1024 * 1024
    elif s.endswith('G'):
        return int(s[:-1]) * 1024 * 1024 * 1024
    return int(s)


class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime.datetime):
            return obj.isoformat()
        return super(DateTimeEncoder, self).default(obj)


# Smart representer: use | only when it makes sense
def smart_str_representer(dumper, data):
    # If the string has newlines OR is longer than ~70 chars → use literal block |
    if '\n' in data or len(data) > 70:
        return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|')
    # For very short strings that are obviously "tags", keep them plain
    if len(data) < 30 and data.isascii() and not data.startswith((' ', '\t')):
        return dumper.represent_scalar('tag:yaml.org,2002:str', data)
    # Otherwise use plain style but allow folding if it's medium length
    return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='')

class PrettyDumper(yaml.SafeDumper):
    pass

PrettyDumper.add_representer(str, smart_str_representer)


def ensureFolder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

def readJson(file):
    with open(file) as f:
        return json.load(f)

def writeJson(file, data):
    with open(file, "w") as f:
        json.dump(data, f, indent=2, cls=DateTimeEncoder)

def readYaml(file):
    with open(file) as f:
        return yaml.safe_load(f)

def parseYaml(text):
    return yaml.safe_load(text)

def writeYaml(file, data):
    with open(file, "w", encoding="utf-8") as f:
        yaml.dump(
            data,
            f,
            indent=2,
            default_flow_style=False,
            allow_unicode=True,
            width=4096,
            Dumper=PrettyDumper,
            sort_keys=False
        )

def readBytes(file):
    with open(file, "rb") as f:
        return f.read()

def writeBytes(file, data):
    with open(file, "wb") as f:
        f.write(data)

def readText(file, encoding="utf-8"):
    for en in ["utf-8", "utf-16", "utf-32", "latin1", "ascii", "iso-8859-1"]:
        try:
            with open(file, "r", encoding=en) as f:
                return f.read()
        except UnicodeDecodeError:
            pass
        except Exception as e:
            pass
    raise Exception(f"Could not read file {file} with any encoding")

def to_utf8(s):
    for en in ["utf-8", "utf-16", "utf-32", "latin1", "ascii", "iso-8859-1"]:
        try:
            return s.encode(en).decode('utf-8')
        except UnicodeDecodeError:
            pass
    return s


def writeText(file, data, encoding="utf-8"):
    ensurePath(file)
    with open(file, "w", encoding=encoding) as f:
        f.write(data)

def writeCsv(filepath, rows, encoding="utf-8"):
    with open(filepath, "w", newline='', encoding=encoding) as f:
        writer = csv.writer(f)
        writer.writerow(rows[0].keys())
        for row in rows:
            writer.writerow(row.values())

def readCsv(filepath, encoding="utf-8"):
    with open(filepath, "r", encoding=encoding) as f:
        reader = csv.reader(f)
        for row in reader:
            yield row

def readCsvDict(filepath, encoding="utf-8"):
    # Return a list of dictionaries.
    with open(filepath, "r", encoding=encoding) as f:
        reader = csv.reader(f)
        keys = reader.__next__()
        for row in reader:
            yield dict(zip(keys, row))


def findPath(relPath, throwIfNotFound=True):
    """
    Recursively search up the folder ancestry for a file of name @relPath.
    """
    fn = os.path.basename(relPath)
    dir = os.path.dirname(relPath)
    
    while True:
        path = os.path.abspath(os.path.join(dir, fn))
        if os.path.exists(path):
            return path
        parent_dir = os.path.abspath(os.path.join(dir, os.pardir))
        # IF dir is the root directory, stop searching (when parent equals current).
        if parent_dir == dir:
            break
        dir = parent_dir
    if throwIfNotFound:
        raise FileNotFoundError(f"Could not find file '{relPath}' in the folder ancestry.")
    return None

def getNewTemporaryFilePath(prefix='', suffix='.txt'):
    """
    Generate a new temporary file path with optional prefix.
    Uses system temp directory and ensures unique filename.
    
    Args:
        prefix (str): Optional prefix for the temp filename
        
    Returns:
        str: Full path to new temporary file
    """
    import tempfile
    import uuid
    
    temp_dir = tempfile.gettempdir()
    unique_name = str(uuid.uuid4())
    if prefix:
        unique_name = f"{prefix}_{unique_name}"
    if suffix:
        unique_name = f"{unique_name}{suffix}"
    temp_path = os.path.join(temp_dir, unique_name)
    
    return temp_path


def getJson(relPath):
    """
    Search up the folder ancestry for a configuration file of name '{relPath}.json'
    and return that as an object.
    @relPath : str : The relative path of the configuration file to search for. Example: "data/config" will search for "data/config.json"
    """
    if not relPath.endswith('.json'):
        relPath += '.json'
    path = findPath(relPath)
    with open(path) as f:
        return json.load(f)

def getYaml(relPath):
    """
    Search up the folder ancestry for a configuration file of name '{relPath}.json'
    and return that as an object.
    Throws FileNotFoundError if the file is not found.
    INCLUDE:
      - person: INCLUDEFILE(data/person.yaml)
    """
    re = getRegex(r"INCLUDEFILE\((?P<A>.+)\)")

    def resolve_includes(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if v:
                    if isinstance(v, str):
                        if 'INCLUDEFILE' in v:
                            pass
                        match = re.match(v)
                        if match:
                            filename = match.group("A")
                            v = readText(filename)
                            o[k] = v
                    else:
                        resolve_includes(v)
        elif isinstance(o, list):
            for v in o:
                resolve_includes(v)

    if not relPath.endswith('.yaml'):
        relPath += '.yaml'
    path = findPath(relPath)    
    yaml = YAML()
    with open(path) as f:
        o = yaml.load(f)
    resolve_includes(o)
    return o


def md5(s): 
  import hashlib
  hash = hashlib.md5()
  if type(s) == str:
    s = str.encode(s)
  hash.update(s)
  return  hash.hexdigest()


def g(o, path, default=None, sep='/'):
    """
    Get a value from a dictionary using a path string.
    Example: g(o, "a/b/c") is equivalent to o['a']['b']['c']
    """
    for k in path.split(sep):
        if o is None:
            return default
        try:
            k = int(k)
        except ValueError:
            if not k in o:
                return default
        o = o[k]
    return o


def gi(o, path, default=None, sep='/'):
    """
    CASE INSENSITIVE
    Get a value from a dictionary using a path string.
    Example: g(o, "a/b/c") is equivalent to o['a']['b']['c']
    """
    for k in path.split(sep):
        if o is None:
            return default
        try:
            k = int(k)
            o = o[k]
        except ValueError:
            found = False
            for K,V in o.items():
                if K.lower() == k.lower():
                    o = V
                    found = True
                    break
            if not found:
                return default
    return o


def lowerKeys(item):
    if item is None:
        return None
    if isinstance(item, dict):
        return {k.lower(): lowerKeys(v) for k,v in item.items()}
    elif isinstance(item, list):
        return [lowerKeys(i) for i in item]
    else:
        return item


def fixJson(j):
    """
    Recursively resolve any value that is a string in the format "/Date(1234567890)/" to a datetime object.
    2147483647
    1729209600000
    Convert string values that are json into objects.
    """
    if isinstance(j, list):
        for i, v in enumerate(j):
            j[i] = fixJson(v)
    elif isinstance(j, dict):
        for k, v in j.items():
            j[k] = fixJson(v)
    elif isinstance(j, str):
        m = re.match(r"\/Date\((-?\d+)\)\/", j)
        if m:
            s = m.group(1)
            n = int(s)
            if n < 0:
                pass
            if len(s) >= 12:
                # Assume milliseconds and convert to seconds
                n = n / 1000
            date = datetime.datetime(1970, 1, 1) + datetime.timedelta(seconds=n)
            return date
        if j.startswith("{") and j.endswith("}"):
            try:
                return json.loads(j)
            except:
                pass
    return j


def ensurePath(path):
    "Guarantees a folder path exists. Creates it if it doesn't."
    if '.' in path:
        if os.path.dirname(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        else:
            pass
    else:
        os.makedirs(path, exist_ok=True)








# ================================================================================
# C A C H E   F U N C T I O N S

# Safe filename, filepath, or url
def clean_filename(s):
    "Get rid of invalid characters"
    invalid_chars = '<>:"?*|'
    for char in invalid_chars:
        s = s.replace(char, "_")
    l = "|"
    while l != s:
        l = s
        s = s.replace("__", "_")
    return s


def get_filename(key):
    "Get the filename for a given key, which is a slash-delimited path."
    key = clean_filename(key)
    path = f"{key}"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def get_file(key):
    try:
        fn = get_filename(key)
        if os.path.exists(fn):
            with open(fn, 'r', encoding='utf-8') as f:
                if fn.endswith('.json'):
                    return json.load(f)
                else:
                    return f.read()
    except Exception as e:
        pass
    return None


def put_file(key, data):
    if isinstance(data, bytes):
        mode = "wb"
    else:
        mode = "w"
    fn = get_filename(key)
    with open(fn, mode) as f:
        if fn.endswith('.json'):
            json.dump(data, f, indent=4, cls=DateTimeEncoder)
        else:
            f.write(data or '')


def get_cache(key):
    if not '.' in key:
        key += '.json'
    return get_file(f"data/cache/{key}")


def put_cache(key, data):
    if not '.' in key:
        key += '.json'
    put_file(f"data/cache/{key}", data)


def cacheGet(fn, func, verbose=True):
    """
    Get the data from the cache if it exists, otherwise call the function to get the data.
    """
    j = get_cache(fn)
    if j:
        print(f"  Cache hit: {fn}")
    else:
        print(f"  Cache miss: {fn}")
        j = func()
        put_cache(fn, j)
    if not j:
        print(f"  Error: {fn}")
    return j








# ================================================================================
# A S S E R T S

def assert_type(o, t, hint=''):
    assert isinstance(o, t), f"Expected type {t}, got {type(o)}. {hint}".strip()
def assert_not_null(o, hint=''):
    assert o != None, f"Expected not null value. {hint}".strip()






# ================================================================================
# P A R Q U E T

def resolve_date_strings(j):
    """
    Recursively resolve any value that is a string in the format "/Date(1234567890)/" to a datetime object.
    2147483647
    1729209600000
    """
    if isinstance(j, list):
        for i, v in enumerate(j):
            j[i] = resolve_date_strings(v)
    elif isinstance(j, dict):
        for k, v in j.items():
            j[k] = resolve_date_strings(v)
    elif isinstance(j, str):
        m = re.match(r"\/Date\((-?\d+)\)\/", j)
        if m:
            s = m.group(1)
            n = int(s)
            if n < 0:
                pass
            if len(s) >= 12:
                # Assume milliseconds and convert to seconds
                n = n / 1000
            date = datetime.datetime(1970, 1, 1) + datetime.timedelta(seconds=n)
            return date
    return j

def to_date(sDate):
    """Convert a string to a date"""
    s = str(sDate)
    if s == None:
        return None
    if s == '':
        return None
    if isinstance(s, datetime.date):
        return s
    re = getRegex(r"\/Date\((?P<A>.+)\)\/")
    match = re.match(s)
    if match:
        s = int(match.group("A"))
        if s <= -62169984000:
            return datetime.datetime.min.isoformat()
        epoch = datetime.datetime(
            1970, 1, 1, tzinfo=datetime.timezone.utc
        )
        dt = epoch + datetime.timedelta(seconds=s)
        s = dt.isoformat().split("T")[0]
    elif "T" in s:
        s = s.split("T")[0]
    elif len(s) == 8:
        s = s[:4] + "-" + s[4:6] + "-" + s[6:]
    else:
        s = s[:10]
    try:
        return datetime.datetime.strptime(s, "%Y-%m-%d").date()
    except Exception as e:
        s = str(sDate)
    for format in [
        "%Y-%m-%d",
        "%m/%d/%Y",
    ]:
        try:
            return datetime.datetime.strptime(s, format)
        except:
            pass



def to_datetime(s):
    """Convert a string to a datetime"""
    if s == None:
        return None
    if s == '':
        return None
    if isinstance(s, datetime.date):
        return s
    re = getRegex(r"\/Date\((?P<A>.+)\)\/")
    match = re.match(s)
    if match:
        s = int(match.group("A"))
        if s <= -62169984000:
            return datetime.datetime.min.isoformat()
        epoch = datetime.datetime(
            1970, 1, 1, tzinfo=datetime.timezone.utc
        )
        dt = epoch + datetime.timedelta(seconds=s)
        s = dt.isoformat().split("T")[0]
    elif "T" in s:
        pass
    elif len(s) == 8:
        s = s[:4] + "-" + s[4:6] + "-" + s[6:]
    else:
        s = s[:10]

    for format in [
        "%Y-%m-%d",
        "%Y-%m-%dT%H:%M:%S.%f%z", 
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "2025-06-24T22:50:22.197Z", 
        "%Y-%m-%dT%H:%M:%SZ", 
        "%Y-%m-%dT%H:%M:%S", 
        "%Y-%m-%dT%H:%M:%S.%f"]:
        try:
            return datetime.datetime.strptime(s, format)
        except:
            pass
    return None


def to_datetime_utc(s):
    """Convert a string to a datetime in UTC"""
    if s == None:
        return None
    if s == '':
        return None
    try:
        return to_datetime(s).astimezone(datetime.timezone.utc)
    except Exception as e:
        pass

    try:
        return to_datetime(s)
    except Exception as e:
        return None
    


def to_int(s):
    if s == None:
        return 0
    return int(s)

def to_float(s):
    if s == None:
        return 0.0
    return float(s)

def is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def is_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

def to_string(o):
    if o == None:
        return ""
    s = str(o)
    if isinstance(o, datetime.datetime):
        return o.isoformat()
    re = getRegex(r"\/Date\((?P<A>.+)\)\/")
    match = re.match(s)
    if match:
        s = int(match.group("A"))
        if s <= -62169984000:
            return datetime.datetime.min.isoformat()
        epoch = datetime.datetime(
            1970, 1, 1, tzinfo=datetime.timezone.utc
        )
        dt = epoch + datetime.timedelta(seconds=s)
        s = dt.isoformat().split("T")[0]
    return s

def dump(obj, level=0, skipNulls=True, header=None):
    "Print out the non-null properties of an object in columnar format."
    if level == 0:
        print()
        header = header or obj.__class__.__name__
        print(header)

    if isinstance(obj, dict) or hasattr(obj, "__data__"):
        if hasattr(obj, "__data__"):
            items = obj.__data__.items()
        else:
            items = obj.items()
        for k,v in sorted(items):
            if k == "birthDate":
                pass
            if skipNulls and v == None:
                continue
            if isinstance(v, dict) or isinstance(v, list):
                print(f"  {'  ' * level}{k + ':'}")
                dump(v, level + 1, skipNulls)
            else:
                v = to_string(v) if v != None else ''
                print(f"  {'  ' * level}{k + ':':<30} {v[:200]}")
    elif isinstance(obj, list):
        if obj:
            for i,v in enumerate(obj):
                if isinstance(v, dict) or isinstance(v, list):
                    print(f"  {'  ' * level}[{i}]")
                    dump(v, level + 1, skipNulls)
                    if len(obj) > 1:
                        print(f"  {'  ' * level}...{len(obj) - i - 1} more")
                    break
                else:
                    print(f"  {'  ' * level}[{i}] = {to_string(v)}")
    else:
        raise Exception(f"Unknown type: {type(obj)}")




# ================================================================================
# R E G E X   F U N C T I O N S

regex = {}  # Cached compiled regex expressions

def getRegex(pattern):
    global regex
    if pattern not in regex:
        regex[pattern] = re.compile(pattern)
    return regex[pattern]

def getPatternRegex(s):
    prefix = r"\b"
    suffix = r"\b"
    if s.endswith("*"):
        s = s[:-1]
        suffix = ""
    if s.startswith("*"):
        s = s[1:]
        prefix = ""
    return getRegex(prefix + re.escape(s) + suffix)




# ================================================================================
# C L A S S E S

class Spy:
    # with Spy() as spy:
    #   ...
    # with Spy('') as spy:
    #   ...
    #   will print out time at end
    def __init__(self, name=None, data=None):
        self.name = name if name != None else datetime.datetime.now().isoformat()
        if self.name != '':
            self.name = f"SPY: {self.name}"
        self.exitHook = None
        self.data = data

    def elapsedSeconds(self):
        return (datetime.datetime.now() - self.start).total_seconds()

    def trace(self, message):
        print(f"  TRACE: {self.elapsedSeconds():.4f}s: {message}")

    def __enter__(self):
        self.start = datetime.datetime.now()
        if self.name != '':
            print(f"SPY START: {self.name}")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.exitHook:
            self.exitHook(self)
        if self.name != '':
            print(f"SPY END:   {self.name} took {self.elapsedSeconds():.4f}s")
