import 'dart:convert';
import 'dart:ui';
import 'package:flutter/material.dart';
import 'package:flutter_map/flutter_map.dart';
import 'package:latlong2/latlong.dart';
import 'package:http/http.dart' as http;

import '../routes.dart';
import '../services/location_service.dart';
import '../widgets/responsive_shell.dart';

class HostelDiscoveryScreen extends StatefulWidget {
  final double initialLat;
  final double initialLng;
  final bool locationEnabled;

  const HostelDiscoveryScreen({
    super.key,
    required this.initialLat,
    required this.initialLng,
    required this.locationEnabled,
  });

  @override
  State<HostelDiscoveryScreen> createState() => _HostelDiscoveryScreenState();
}

class _HostelDiscoveryScreenState extends State<HostelDiscoveryScreen>
    with TickerProviderStateMixin {
  final MapController _mapController = MapController();
  final TextEditingController _searchController = TextEditingController();

  late bool _locationEnabled;
  late LatLng _currentCenter;
  double _zoom = 14;

  bool _loadingHostels = false;
  final List<_HostelData> _hostels = [];
  Marker? _searchMarker;

  // Bottom sheet animation
  late AnimationController _sheetController;
  late Animation<double> _sheetAnimation;
  bool _sheetExpanded = false;

  // Selected filter chip
  String? _selectedFilter;

  static const teal = Color(0xFF0B7C80);
  static const tealDark = Color(0xFF085F62);

  final List<Map<String, dynamic>> _filters = [
    {'label': 'Distance', 'icon': Icons.near_me},
    {'label': 'AI Recommend', 'icon': Icons.auto_awesome},
    {'label': 'Hostel Type', 'icon': Icons.home_work},
    {'label': 'Budget', 'icon': Icons.attach_money},
    {'label': 'Amenities', 'icon': Icons.star},
    {'label': 'Stay', 'icon': Icons.king_bed},
    {'label': 'Food', 'icon': Icons.restaurant},
    {'label': 'Room Type', 'icon': Icons.meeting_room},
    {'label': 'Rating & Reviews', 'icon': Icons.reviews},
  ];

  @override
  void initState() {
    super.initState();
    _locationEnabled = widget.locationEnabled;
    _currentCenter = LatLng(widget.initialLat, widget.initialLng);
    _loadNearbyHostels(_currentCenter);

    _sheetController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 350),
    );
    _sheetAnimation = CurvedAnimation(
      parent: _sheetController,
      curve: Curves.easeOutCubic,
    );
  }

  @override
  void dispose() {
    _searchController.dispose();
    _sheetController.dispose();
    super.dispose();
  }

  Future<void> _searchPlace(String query) async {
    final q = query.trim();
    if (q.isEmpty) return;

    final uri = Uri.parse(
      "https://nominatim.openstreetmap.org/search?q=${Uri.encodeComponent(q)}&format=json&limit=1",
    );

    try {
      final res = await http.get(uri, headers: {"User-Agent": "staybuddy_student_app"});
      if (res.statusCode != 200) return;
      final data = jsonDecode(res.body);
      if (data is! List || data.isEmpty) return;
      final lat = double.tryParse(data[0]["lat"] ?? "");
      final lon = double.tryParse(data[0]["lon"] ?? "");
      if (lat == null || lon == null) return;
      final found = LatLng(lat, lon);
      setState(() {
        _currentCenter = found;
        _zoom = 15;
        _searchMarker = Marker(
          point: found,
          width: 50,
          height: 50,
          child: const Icon(Icons.location_pin, size: 44, color: Colors.red),
        );
      });
      _mapController.move(found, _zoom);
      await _loadNearbyHostels(found);
    } catch (_) {}
  }

  Future<void> _loadNearbyHostels(LatLng center) async {
    setState(() => _loadingHostels = true);
    const radius = 2500;
    final query = """
[out:json];
(
  node["tourism"="hostel"](around:$radius,${center.latitude},${center.longitude});
  way["tourism"="hostel"](around:$radius,${center.latitude},${center.longitude});
  relation["tourism"="hostel"](around:$radius,${center.latitude},${center.longitude});
);
out center;
""";

    try {
      final uri = Uri.parse("https://overpass-api.de/api/interpreter");
      final res = await http.post(
        uri,
        headers: {"Content-Type": "application/x-www-form-urlencoded"},
        body: {"data": query},
      );
      if (res.statusCode != 200) {
        setState(() => _loadingHostels = false);
        return;
      }
      final json = jsonDecode(res.body);
      final elements = (json["elements"] as List?) ?? [];
      final hostels = <_HostelData>[];
      for (final el in elements) {
        double? lat;
        double? lon;
        if (el["type"] == "node") {
          lat = (el["lat"] as num?)?.toDouble();
          lon = (el["lon"] as num?)?.toDouble();
        } else {
          final c = el["center"];
          lat = (c?["lat"] as num?)?.toDouble();
          lon = (c?["lon"] as num?)?.toDouble();
        }
        if (lat == null || lon == null) continue;
        final name = (el["tags"]?["name"] ?? "Hostel").toString();
        hostels.add(_HostelData(name: name, point: LatLng(lat, lon)));
      }
      setState(() {
        _hostels
          ..clear()
          ..addAll(hostels);
        _loadingHostels = false;
      });
    } catch (_) {
      setState(() => _loadingHostels = false);
    }
  }

  Future<void> _goToMyLocation() async {
    final pos = await LocationService.getCurrentPosition();
    if (pos == null) return;
    final here = LatLng(pos.latitude, pos.longitude);
    setState(() {
      _locationEnabled = true;
      _currentCenter = here;
      _zoom = 15;
    });
    _mapController.move(here, _zoom);
    await _loadNearbyHostels(here);
  }

  void _zoomIn() {
    setState(() => _zoom = (_zoom + 1).clamp(3, 19));
    _mapController.move(_currentCenter, _zoom);
  }

  void _zoomOut() {
    setState(() => _zoom = (_zoom - 1).clamp(3, 19));
    _mapController.move(_currentCenter, _zoom);
  }

  void _toggleSheet() {
    setState(() => _sheetExpanded = !_sheetExpanded);
    if (_sheetExpanded) {
      _sheetController.forward();
    } else {
      _sheetController.reverse();
    }
  }

  @override
  Widget build(BuildContext context) {
    final size = MediaQuery.of(context).size;
    final markers = <Marker>[
      // User location marker
      Marker(
        point: _currentCenter,
        width: 56,
        height: 56,
        child: Container(
          decoration: BoxDecoration(
            shape: BoxShape.circle,
            color: Colors.blue.shade600,
            border: Border.all(color: Colors.white, width: 3),
            boxShadow: [BoxShadow(color: Colors.blue.shade200, blurRadius: 12, spreadRadius: 2)],
          ),
          child: const Icon(Icons.person, color: Colors.white, size: 22),
        ),
      ),
      if (_searchMarker != null) _searchMarker!,
      ..._hostels.map((h) => Marker(
            point: h.point,
            width: 50,
            height: 50,
            child: Tooltip(
              message: h.name,
              child: Container(
                decoration: BoxDecoration(
                  color: teal,
                  shape: BoxShape.circle,
                  border: Border.all(color: Colors.white, width: 2.5),
                  boxShadow: [BoxShadow(color: teal.withOpacity(0.4), blurRadius: 8, spreadRadius: 1)],
                ),
                child: const Icon(Icons.home, color: Colors.white, size: 22),
              ),
            ),
          )),
    ];

    return ResponsiveShell(
      child: Scaffold(
      body: Stack(
        children: [
          // ── MAP ──────────────────────────────────────────────
          FlutterMap(
            mapController: _mapController,
            options: MapOptions(
              initialCenter: _currentCenter,
              initialZoom: _zoom,
            ),
            children: [
              TileLayer(
                urlTemplate: "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
                userAgentPackageName: "staybuddy_student",
              ),
              MarkerLayer(markers: markers),
            ],
          ),

          // ── TOP HEADER ──────────────────────────────────────
          Positioned(
            left: 0,
            right: 0,
            top: 0,
            child: ClipRRect(
              child: BackdropFilter(
                filter: ImageFilter.blur(sigmaX: 18, sigmaY: 18),
                child: Container(
                  decoration: BoxDecoration(
                    gradient: LinearGradient(
                      begin: Alignment.topLeft,
                      end: Alignment.bottomRight,
                      colors: [
                        teal.withOpacity(0.92),
                        tealDark.withOpacity(0.88),
                      ],
                    ),
                  ),
                  child: SafeArea(
                    bottom: false,
                    child: Padding(
                      padding: const EdgeInsets.fromLTRB(16, 12, 16, 16),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          // Logo row + back
                          Row(
                            children: [
                              GestureDetector(
                                onTap: () => Navigator.pushReplacementNamed(
                                    context, Routes.locationAccess),
                                child: Container(
                                  width: 38,
                                  height: 38,
                                  decoration: BoxDecoration(
                                    color: Colors.white.withOpacity(0.2),
                                    shape: BoxShape.circle,
                                    border: Border.all(
                                        color: Colors.white.withOpacity(0.3)),
                                  ),
                                  child: const Icon(Icons.arrow_back,
                                      color: Colors.white, size: 18),
                                ),
                              ),
                              const SizedBox(width: 12),
                              Image.asset(
                                "assets/images/logo.png",
                                width: 36,
                                height: 36,
                                fit: BoxFit.contain,
                              ),
                              const SizedBox(width: 10),
                              Column(
                                crossAxisAlignment: CrossAxisAlignment.start,
                                children: [
                                  const Text(
                                    "StayBuddy",
                                    style: TextStyle(
                                      color: Colors.white,
                                      fontSize: 18,
                                      fontWeight: FontWeight.w800,
                                      letterSpacing: 0.5,
                                    ),
                                  ),
                                  Text(
                                    "Hostel Discovery",
                                    style: TextStyle(
                                      color: Colors.white.withOpacity(0.75),
                                      fontSize: 12,
                                      fontWeight: FontWeight.w500,
                                    ),
                                  ),
                                ],
                              ),
                              const Spacer(),
                              // Hostel count badge
                              if (_hostels.isNotEmpty)
                                Container(
                                  padding: const EdgeInsets.symmetric(
                                      horizontal: 10, vertical: 5),
                                  decoration: BoxDecoration(
                                    color: Colors.white.withOpacity(0.2),
                                    borderRadius: BorderRadius.circular(20),
                                    border: Border.all(
                                        color: Colors.white.withOpacity(0.3)),
                                  ),
                                  child: Text(
                                    "${_hostels.length} Hostels",
                                    style: const TextStyle(
                                      color: Colors.white,
                                      fontSize: 12,
                                      fontWeight: FontWeight.w700,
                                    ),
                                  ),
                                ),
                            ],
                          ),
                          const SizedBox(height: 14),

                          // ── SEARCH BAR ─────────────────────
                          ClipRRect(
                            borderRadius: BorderRadius.circular(28),
                            child: BackdropFilter(
                              filter: ImageFilter.blur(sigmaX: 10, sigmaY: 10),
                              child: Container(
                                height: 48,
                                decoration: BoxDecoration(
                                  color: Colors.white.withOpacity(0.18),
                                  borderRadius: BorderRadius.circular(28),
                                  border: Border.all(
                                      color: Colors.white.withOpacity(0.35)),
                                ),
                                child: Row(
                                  children: [
                                    const SizedBox(width: 16),
                                    Icon(Icons.search,
                                        color: Colors.white.withOpacity(0.9),
                                        size: 20),
                                    const SizedBox(width: 10),
                                    Expanded(
                                      child: TextField(
                                        controller: _searchController,
                                        onSubmitted: _searchPlace,
                                        style: const TextStyle(
                                            color: Colors.white, fontSize: 14),
                                        decoration: InputDecoration(
                                          border: InputBorder.none,
                                          hintText:
                                              "Search by city, university or area...",
                                          hintStyle: TextStyle(
                                            color:
                                                Colors.white.withOpacity(0.6),
                                            fontSize: 13,
                                          ),
                                          filled: false,
                                        ),
                                      ),
                                    ),
                                    GestureDetector(
                                      onTap: () =>
                                          _searchPlace(_searchController.text),
                                      child: Container(
                                        width: 48,
                                        height: 48,
                                        decoration: BoxDecoration(
                                          color: Colors.white.withOpacity(0.2),
                                          borderRadius: const BorderRadius.only(
                                            topRight: Radius.circular(28),
                                            bottomRight: Radius.circular(28),
                                          ),
                                        ),
                                        child: const Icon(Icons.search,
                                            color: Colors.white, size: 20),
                                      ),
                                    ),
                                  ],
                                ),
                              ),
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
                ),
              ),
            ),
          ),

          // ── LOCATION DISABLED BANNER ─────────────────────────
          if (!widget.locationEnabled && !_locationEnabled)
            Positioned(
              left: 16,
              right: 16,
              top: 200,
              child: ClipRRect(
                borderRadius: BorderRadius.circular(16),
                child: BackdropFilter(
                  filter: ImageFilter.blur(sigmaX: 10, sigmaY: 10),
                  child: Container(
                    padding: const EdgeInsets.symmetric(
                        horizontal: 16, vertical: 12),
                    decoration: BoxDecoration(
                      color: Colors.orange.shade800.withOpacity(0.85),
                      borderRadius: BorderRadius.circular(16),
                      border: Border.all(
                          color: Colors.orange.shade300.withOpacity(0.4)),
                    ),
                    child: Row(
                      children: [
                        const Icon(Icons.location_off,
                            color: Colors.white, size: 20),
                        const SizedBox(width: 10),
                        const Expanded(
                          child: Text(
                            "Location is off. Enable to see nearby hostels.",
                            style: TextStyle(
                                color: Colors.white,
                                fontSize: 13,
                                fontWeight: FontWeight.w500),
                          ),
                        ),
                        GestureDetector(
                          onTap: _goToMyLocation,
                          child: Container(
                            padding: const EdgeInsets.symmetric(
                                horizontal: 12, vertical: 6),
                            decoration: BoxDecoration(
                              color: Colors.white,
                              borderRadius: BorderRadius.circular(20),
                            ),
                            child: const Text(
                              "Enable",
                              style: TextStyle(
                                color: Colors.deepOrange,
                                fontWeight: FontWeight.w700,
                                fontSize: 12,
                              ),
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
              ),
            ),

          // ── RIGHT SIDE FABs ──────────────────────────────────
          Positioned(
            right: 16,
            bottom: _sheetExpanded ? 320 : 220,
            child: Column(
              children: [
                _GlassFab(icon: Icons.add, onTap: _zoomIn),
                const SizedBox(height: 10),
                _GlassFab(icon: Icons.remove, onTap: _zoomOut),
                const SizedBox(height: 10),
                _GlassFab(
                  icon: _locationEnabled
                      ? Icons.my_location
                      : Icons.location_disabled,
                  onTap: _goToMyLocation,
                  color: _locationEnabled ? teal : Colors.grey,
                ),
              ],
            ),
          ),

          // ── LOADING INDICATOR ────────────────────────────────
          if (_loadingHostels)
            Positioned(
              left: 16,
              right: 16,
              bottom: _sheetExpanded ? 310 : 210,
              child: Center(
                child: ClipRRect(
                  borderRadius: BorderRadius.circular(30),
                  child: BackdropFilter(
                    filter: ImageFilter.blur(sigmaX: 10, sigmaY: 10),
                    child: Container(
                      padding: const EdgeInsets.symmetric(
                          horizontal: 18, vertical: 10),
                      decoration: BoxDecoration(
                        color: teal.withOpacity(0.85),
                        borderRadius: BorderRadius.circular(30),
                      ),
                      child: const Row(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          SizedBox(
                            width: 14,
                            height: 14,
                            child: CircularProgressIndicator(
                                strokeWidth: 2, color: Colors.white),
                          ),
                          SizedBox(width: 10),
                          Text("Finding nearby hostels...",
                              style: TextStyle(
                                  color: Colors.white,
                                  fontSize: 13,
                                  fontWeight: FontWeight.w600)),
                        ],
                      ),
                    ),
                  ),
                ),
              ),
            ),

          // ── BOTTOM SHEET ─────────────────────────────────────
          Positioned(
            left: 0,
            right: 0,
            bottom: 0,
            child: GestureDetector(
              onVerticalDragEnd: (d) {
                if (d.primaryVelocity! < -300 && !_sheetExpanded) _toggleSheet();
                if (d.primaryVelocity! > 300 && _sheetExpanded) _toggleSheet();
              },
              child: AnimatedContainer(
                duration: const Duration(milliseconds: 350),
                curve: Curves.easeOutCubic,
                decoration: BoxDecoration(
                  gradient: LinearGradient(
                    begin: Alignment.topLeft,
                    end: Alignment.bottomRight,
                    colors: [
                      teal.withOpacity(0.97),
                      tealDark.withOpacity(0.97),
                    ],
                  ),
                  borderRadius: const BorderRadius.only(
                    topLeft: Radius.circular(32),
                    topRight: Radius.circular(32),
                  ),
                  boxShadow: [
                    BoxShadow(
                      color: teal.withOpacity(0.35),
                      blurRadius: 30,
                      offset: const Offset(0, -6),
                    ),
                  ],
                ),
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    // drag handle
                    GestureDetector(
                      onTap: _toggleSheet,
                      child: Padding(
                        padding: const EdgeInsets.only(top: 12, bottom: 4),
                        child: Container(
                          width: 44,
                          height: 4,
                          decoration: BoxDecoration(
                            color: Colors.white.withOpacity(0.4),
                            borderRadius: BorderRadius.circular(4),
                          ),
                        ),
                      ),
                    ),

                    Padding(
                      padding: const EdgeInsets.fromLTRB(20, 8, 20, 20),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          // Title row
                          Row(
                            children: [
                              Container(
                                padding: const EdgeInsets.all(8),
                                decoration: BoxDecoration(
                                  color: Colors.white.withOpacity(0.15),
                                  borderRadius: BorderRadius.circular(12),
                                ),
                                child: const Icon(Icons.home_work_outlined,
                                    color: Colors.white, size: 20),
                              ),
                              const SizedBox(width: 12),
                              const Text(
                                "Discover Hostels",
                                style: TextStyle(
                                  color: Colors.white,
                                  fontSize: 20,
                                  fontWeight: FontWeight.w800,
                                  letterSpacing: 0.3,
                                ),
                              ),
                              const Spacer(),
                              if (_hostels.isNotEmpty)
                                Container(
                                  padding: const EdgeInsets.symmetric(
                                      horizontal: 10, vertical: 4),
                                  decoration: BoxDecoration(
                                    color: Colors.white.withOpacity(0.15),
                                    borderRadius: BorderRadius.circular(20),
                                    border: Border.all(
                                        color: Colors.white.withOpacity(0.25)),
                                  ),
                                  child: Text(
                                    "Showing ${_hostels.length}",
                                    style: const TextStyle(
                                      color: Colors.white,
                                      fontSize: 12,
                                      fontWeight: FontWeight.w700,
                                    ),
                                  ),
                                ),
                            ],
                          ),
                          const SizedBox(height: 16),

                          // Filter chips
                          Text(
                            "Filter by",
                            style: TextStyle(
                              color: Colors.white.withOpacity(0.7),
                              fontSize: 12,
                              fontWeight: FontWeight.w600,
                              letterSpacing: 0.8,
                            ),
                          ),
                          const SizedBox(height: 10),
                          SizedBox(
                            height: 40,
                            child: ListView.separated(
                              scrollDirection: Axis.horizontal,
                              itemCount: _filters.length,
                              separatorBuilder: (_, __) =>
                                  const SizedBox(width: 8),
                              itemBuilder: (ctx, i) {
                                final f = _filters[i];
                                final selected =
                                    _selectedFilter == f['label'];
                                return GestureDetector(
                                  onTap: () => setState(() =>
                                      _selectedFilter = selected
                                          ? null
                                          : f['label'] as String),
                                  child: AnimatedContainer(
                                    duration:
                                        const Duration(milliseconds: 200),
                                    padding: const EdgeInsets.symmetric(
                                        horizontal: 14, vertical: 8),
                                    decoration: BoxDecoration(
                                      color: selected
                                          ? Colors.white
                                          : Colors.white.withOpacity(0.15),
                                      borderRadius:
                                          BorderRadius.circular(20),
                                      border: Border.all(
                                        color: selected
                                            ? Colors.white
                                            : Colors.white.withOpacity(0.3),
                                      ),
                                    ),
                                    child: Row(
                                      mainAxisSize: MainAxisSize.min,
                                      children: [
                                        Icon(
                                          f['icon'] as IconData,
                                          size: 14,
                                          color: selected ? teal : Colors.white,
                                        ),
                                        const SizedBox(width: 6),
                                        Text(
                                          f['label'] as String,
                                          style: TextStyle(
                                            color: selected
                                                ? teal
                                                : Colors.white,
                                            fontSize: 12,
                                            fontWeight: FontWeight.w700,
                                          ),
                                        ),
                                      ],
                                    ),
                                  ),
                                );
                              },
                            ),
                          ),
                          const SizedBox(height: 14),

                          // AI Recommend button
                          SizedBox(
                            width: double.infinity,
                            height: 50,
                            child: ElevatedButton.icon(
                              style: ElevatedButton.styleFrom(
                                backgroundColor: Colors.white,
                                foregroundColor: teal,
                                elevation: 0,
                                shape: RoundedRectangleBorder(
                                  borderRadius: BorderRadius.circular(28),
                                ),
                              ),
                              onPressed: () {},
                              icon: const Icon(Icons.auto_awesome, size: 18),
                              label: const Text(
                                "AI Recommendation",
                                style: TextStyle(
                                  fontSize: 15,
                                  fontWeight: FontWeight.w800,
                                ),
                              ),
                            ),
                          ),
                        ],
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ),
        ],
      ),
    ),
    );
  }
}

class _HostelData {
  final String name;
  final LatLng point;
  _HostelData({required this.name, required this.point});
}

class _GlassFab extends StatelessWidget {
  final IconData icon;
  final VoidCallback onTap;
  final Color? color;
  const _GlassFab({required this.icon, required this.onTap, this.color});

  @override
  Widget build(BuildContext context) {
    const teal = Color(0xFF0B7C80);
    return GestureDetector(
      onTap: onTap,
      child: ClipRRect(
        borderRadius: BorderRadius.circular(999),
        child: BackdropFilter(
          filter: ImageFilter.blur(sigmaX: 10, sigmaY: 10),
          child: Container(
            width: 48,
            height: 48,
            decoration: BoxDecoration(
              color: (color ?? teal).withOpacity(0.9),
              shape: BoxShape.circle,
              border:
                  Border.all(color: Colors.white.withOpacity(0.3), width: 1.5),
              boxShadow: [
                BoxShadow(
                  color: (color ?? teal).withOpacity(0.4),
                  blurRadius: 12,
                  spreadRadius: 1,
                ),
              ],
            ),
            child: Icon(icon, color: Colors.white, size: 22),
          ),
        ),
      ),
    );
  }
}