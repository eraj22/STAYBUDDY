import 'dart:async';
import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:flutter_map/flutter_map.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:http/http.dart' as http;
import 'package:latlong2/latlong.dart';

import '../theme.dart';
import '../widgets/responsive_shell.dart';
import '../models/owner_attendance_boundary_draft.dart';
import 'owner_floor_room_setup_screen.dart';

class OwnerHostelMapScreen extends StatefulWidget {
  final String ownerEmail;
  final String ownerName;
  final String hostelName;
  final String hostelType;
  final String city;
  final String area;
  final String address;

  const OwnerHostelMapScreen({
    super.key,
    required this.ownerEmail,
    required this.ownerName,
    required this.hostelName,
    required this.hostelType,
    required this.city,
    this.area = '',
    required this.address,
  });

  @override
  State<OwnerHostelMapScreen> createState() => _OwnerHostelMapScreenState();
}

class _OwnerHostelMapScreenState extends State<OwnerHostelMapScreen>
    with SingleTickerProviderStateMixin {
  final MapController _mapCtrl = MapController();

  final TextEditingController _searchCtrl = TextEditingController();

  final FocusNode _searchFocus = FocusNode();

  // Default Islamabad
  LatLng _pinnedLocation = const LatLng(33.6844, 73.0479);

  bool _pinSet = false;
  bool _searching = false;
  int _searchRequest = 0;
  Timer? _searchDebounce;
  bool _drawingBoundary = false;
  final List<LatLng> _attendanceBoundary = [];

  List<Map<String, dynamic>> _suggestions = [];

  bool _showSuggestions = false;

  late final AnimationController _panelAnim;

  late final Animation<double> _panelSlide;

  late final Animation<double> _panelFade;

  static const Map<String, LatLng> _cityCenters = {
    'islamabad': LatLng(33.6844, 73.0479),
    'rawalpindi': LatLng(33.5651, 73.0169),
    'lahore': LatLng(31.5204, 74.3587),
    'karachi': LatLng(24.8607, 67.0011),
    'faisalabad': LatLng(31.4504, 73.1350),
    'multan': LatLng(30.1575, 71.5249),
    'peshawar': LatLng(34.0151, 71.5249),
    'quetta': LatLng(30.1798, 66.9750),
    'gujranwala': LatLng(32.1877, 74.1945),
    'sialkot': LatLng(32.4945, 74.5229),
    'hyderabad': LatLng(25.3960, 68.3578),
    'bahawalpur': LatLng(29.3956, 71.6836),
    'sukkur': LatLng(27.7052, 68.8574),
    'abbottabad': LatLng(34.1463, 73.2117),
    'gilgit': LatLng(35.9208, 74.3144),
    'skardu': LatLng(35.2971, 75.6337),
    'muzaffarabad': LatLng(34.3700, 73.4711),
  };

  static const Map<String, LatLng> _areaCenters = {
    'islamabad|h-11': LatLng(33.6543909, 73.0091541),
    'islamabad|h-10': LatLng(33.6629, 73.0037),
    'islamabad|h-9': LatLng(33.6687, 73.0214),
    'islamabad|h-8': LatLng(33.6772, 73.0351),
    'islamabad|i-8': LatLng(33.6541, 73.0721),
    'islamabad|i-9': LatLng(33.6415, 73.0732),
    'islamabad|i-10': LatLng(33.6320, 73.0667),
    'islamabad|i-11': LatLng(33.6208, 73.0492),
  };

  // ===========================================================================
  // LIFECYCLE
  // ===========================================================================

  @override
  void initState() {
    super.initState();

    OwnerAttendanceBoundaryDraft.clear();

    final cityKey = widget.city.trim().toLowerCase();
    final areaKey = widget.area.trim().toLowerCase();
    _pinnedLocation =
        _areaCenters['$cityKey|$areaKey'] ??
        _cityCenters[cityKey] ??
        _pinnedLocation;

    _panelAnim = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 650),
    );

    _panelSlide = Tween<double>(
      begin: 45,
      end: 0,
    ).animate(CurvedAnimation(parent: _panelAnim, curve: Curves.easeOutCubic));

    _panelFade = CurvedAnimation(parent: _panelAnim, curve: Curves.easeOut);

    _panelAnim.forward();

    // Carry the owner-entered location into the map and look it up immediately.
    // The owner can still adjust the final pin manually if a street-level match
    // is unavailable in OpenStreetMap.
    _searchCtrl.text = _addressQuery;
    WidgetsBinding.instance.addPostFrameCallback((_) {
      _search(_addressQuery);
    });
  }

  String get _addressQuery {
    final pieces = <String>[
      widget.address.trim(),
      widget.city.trim(),
      'Pakistan',
    ]..removeWhere((piece) => piece.isEmpty);
    return pieces.join(', ');
  }

  @override
  void dispose() {
    _searchDebounce?.cancel();
    _panelAnim.dispose();

    _searchCtrl.dispose();
    _searchFocus.dispose();

    super.dispose();
  }

  // ===========================================================================
  // SEARCH
  // ===========================================================================

  void _queueSearch(String query) {
    _searchDebounce?.cancel();
    _searchDebounce = Timer(const Duration(milliseconds: 550), () {
      _search(query);
    });
  }

  Future<void> _search(String query) async {
    final q = query.trim();
    final searchRequest = ++_searchRequest;

    if (q.isEmpty) {
      if (!mounted) return;

      setState(() {
        _suggestions = [];
        _showSuggestions = false;
        _searching = false;
      });

      return;
    }

    setState(() {
      _searching = true;
    });

    try {
      final city = widget.city.trim();
      final searchQuery = q.toLowerCase().contains(city.toLowerCase())
          ? q
          : '$q, $city, Pakistan';
      // Always try the full house/street address first.  Previously G-13 was
      // searched before "G-13/4 Street 130 House 26", so a sector-centre
      // result could hide the more precise address from the owner.
      final sectorMatch = RegExp(r'\b[A-Za-z]-?\d{1,2}(?:\s*/\s*\d{1,2})?\b')
          .firstMatch(q);
      final searchQueries = <String>[
        searchQuery,
        if (sectorMatch != null) '${sectorMatch.group(0)}, $city, Pakistan',
      ];

      final decoded = <dynamic>[];
      for (final candidate in searchQueries) {
        final uri = Uri.parse(
          'https://nominatim.openstreetmap.org/search'
          '?q=${Uri.encodeComponent(candidate)}'
          '&format=jsonv2'
          '&addressdetails=1'
          '&countrycodes=pk'
          '&limit=8',
        );

        final response = await http.get(
          uri,
          headers: {'User-Agent': 'staybuddy_owner'},
        );

        if (!mounted || searchRequest != _searchRequest) return;

        if (response.statusCode < 200 || response.statusCode >= 300) {
          continue;
        }

        final payload = jsonDecode(response.body);
        if (payload is List && payload.isNotEmpty) {
          decoded.addAll(payload);
        }
      }

      if (decoded.isEmpty) {
        throw Exception('Invalid location response');
      }

      final results = decoded
          .whereType<Map>()
          .map<Map<String, dynamic>>((item) {
            return {
              'name': item['display_name']?.toString() ?? '',
              'lat': double.tryParse(item['lat']?.toString() ?? ''),
              'lng': double.tryParse(item['lon']?.toString() ?? ''),
            };
          })
          .where((item) => item['lat'] is double && item['lng'] is double)
          .fold<List<Map<String, dynamic>>>([], (unique, item) {
            final exists = unique.any(
              (saved) =>
                  saved['lat'] == item['lat'] && saved['lng'] == item['lng'],
            );
            if (!exists) unique.add(item);
            return unique;
          });

      setState(() {
        _suggestions = results;
        _showSuggestions = results.isNotEmpty;
        _searching = false;
      });
    } catch (error) {
      debugPrint('Map search error: $error');

      if (!mounted || searchRequest != _searchRequest) return;

      setState(() {
        _searching = false;
        _showSuggestions = false;
      });
    }
  }

  void _selectSuggestion(Map<String, dynamic> suggestion) {
    final latitude = suggestion['lat'] as double;

    final longitude = suggestion['lng'] as double;

    final location = LatLng(latitude, longitude);

    final displayName = suggestion['name'].toString();

    setState(() {
      _pinnedLocation = location;

      _pinSet = true;

      _showSuggestions = false;

      _searchCtrl.text = displayName.split(',').first.trim();
    });

    _searchFocus.unfocus();

    // A search result is only a starting point: house-level map data is not
    // available for every Islamabad address.  High zoom lets the owner place
    // the final pin on the actual hostel building in satellite imagery.
    _mapCtrl.move(location, 18.5);
    _showMobileMessage(
      'Zoom in, then tap the exact hostel building to confirm the final pin.',
    );
  }

  // ===========================================================================
  // MAP
  // ===========================================================================

  void _onMapTap(TapPosition tapPosition, LatLng location) {
    if (_drawingBoundary) {
      setState(() {
        _attendanceBoundary.add(location);
        _pinnedLocation = location;
        _pinSet = true;
        _showSuggestions = false;
      });
      return;
    }

    setState(() {
      _pinnedLocation = location;

      _pinSet = true;

      _showSuggestions = false;
    });

    FocusScope.of(context).unfocus();
  }

  void _zoomIn() {
    _mapCtrl.move(
      _mapCtrl.camera.center,
      (_mapCtrl.camera.zoom + 1).clamp(2, 19),
    );
  }

  void _zoomOut() {
    _mapCtrl.move(
      _mapCtrl.camera.center,
      (_mapCtrl.camera.zoom - 1).clamp(2, 19),
    );
  }

  // ===========================================================================
  // NEXT
  // ===========================================================================

  void _next() {
    if (_attendanceBoundary.length < 3) {
      _showMobileMessage(
        'Mark at least 3 points around the hostel building, gate, and entrance.',
        error: true,
      );
      return;
    }

    if (!_pinSet) {
      _showMobileMessage('Please pin your hostel location first.', error: true);

      return;
    }

    OwnerAttendanceBoundaryDraft.replace(
      _attendanceBoundary
          .map(
            (point) => {
              'latitude': point.latitude,
              'longitude': point.longitude,
            },
          )
          .toList(growable: false),
    );

    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (_) => OwnerFloorRoomSetupScreen(
          ownerEmail: widget.ownerEmail,
          ownerName: widget.ownerName,
          hostelName: widget.hostelName,
          hostelType: widget.hostelType,
          city: widget.city,
          address: widget.address,
          latitude: _pinnedLocation.latitude,
          longitude: _pinnedLocation.longitude,
        ),
      ),
    );
  }

  // ===========================================================================
  // MOBILE MESSAGE
  // ===========================================================================

  void _showMobileMessage(String message, {bool error = false}) {
    final messenger = ScaffoldMessenger.maybeOf(context);

    if (messenger == null) {
      return;
    }

    messenger
      ..hideCurrentSnackBar()
      ..showSnackBar(
        SnackBar(
          behavior: SnackBarBehavior.floating,
          elevation: 0,
          margin: const EdgeInsets.fromLTRB(18, 0, 18, 18),
          padding: EdgeInsets.zero,
          backgroundColor: Colors.transparent,
          duration: const Duration(seconds: 2),
          content: Container(
            padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
            decoration: BoxDecoration(
              color: error ? const Color(0xFF9B3838) : AppTheme.teal,
              borderRadius: BorderRadius.circular(14),
              boxShadow: [
                BoxShadow(
                  color: Colors.black.withOpacity(0.24),
                  blurRadius: 18,
                  offset: const Offset(0, 7),
                ),
              ],
            ),
            child: Row(
              children: [
                Icon(
                  error
                      ? Icons.error_outline_rounded
                      : Icons.check_circle_outline_rounded,
                  color: Colors.white,
                  size: 18,
                ),
                const SizedBox(width: 9),
                Expanded(
                  child: Text(
                    message,
                    style: GoogleFonts.dmSans(
                      color: Colors.white,
                      fontSize: 10.5,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                ),
              ],
            ),
          ),
        ),
      );
  }

  // ===========================================================================
  // BUILD
  // ===========================================================================

  @override
  Widget build(BuildContext context) {
    return ResponsiveShell(
      child: Scaffold(
        resizeToAvoidBottomInset: false,
        backgroundColor: const Color(0xFF021F22),
        body: GestureDetector(
          behavior: HitTestBehavior.translucent,
          onTap: () {
            FocusScope.of(context).unfocus();

            if (_showSuggestions) {
              setState(() {
                _showSuggestions = false;
              });
            }
          },
          child: LayoutBuilder(
            builder: (context, constraints) {
              final safeTop = MediaQuery.paddingOf(context).top;

              final safeBottom = MediaQuery.paddingOf(context).bottom;

              // Responsive panel size.
              const panelHeight = 236.0;

              return Stack(
                clipBehavior: Clip.none,
                children: [
                  // ===========================================================
                  // MAP
                  // ===========================================================
                  Positioned.fill(
                    child: FlutterMap(
                      mapController: _mapCtrl,
                      options: MapOptions(
                        initialCenter: _pinnedLocation,
                        initialZoom: 17,
                        minZoom: 2,
                        maxZoom: 19,
                        onTap: _onMapTap,
                      ),
                      children: [
                        TileLayer(
                          urlTemplate:
                              'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
                          userAgentPackageName: 'staybuddy_owner',
                          maxNativeZoom: 19,
                        ),

                        TileLayer(
                          urlTemplate:
                              'https://services.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}',
                          userAgentPackageName: 'staybuddy_owner',
                          maxNativeZoom: 19,
                        ),

                        const RichAttributionWidget(
                          attributions: [
                            TextSourceAttribution(
                              'Esri, Maxar, Earthstar Geographics',
                            ),
                          ],
                        ),

                        if (_attendanceBoundary.length >= 3)
                          PolygonLayer(
                            polygons: [
                              Polygon(
                                points: _attendanceBoundary,
                                color: AppTheme.tealMint.withOpacity(0.24),
                                borderColor: const Color(0xFFFFB454),
                                borderStrokeWidth: 3,
                              ),
                            ],
                          ),

                        MarkerLayer(
                          markers: [
                            if (_pinSet)
                              Marker(
                                point: _pinnedLocation,
                                width: 62,
                                height: 74,
                                child: _buildMapMarker(),
                              ),
                            ..._attendanceBoundary.map(
                              (point) => Marker(
                                point: point,
                                width: 16,
                                height: 16,
                                child: Container(
                                  decoration: BoxDecoration(
                                    color: const Color(0xFFFFB454),
                                    shape: BoxShape.circle,
                                    border: Border.all(
                                      color: Colors.white,
                                      width: 2,
                                    ),
                                  ),
                                ),
                              ),
                            ),
                          ],
                        ),
                      ],
                    ),
                  ),

                  // Slight cinematic map tint.
                  Positioned.fill(
                    child: IgnorePointer(
                      child: Container(
                        color: const Color(0xFF002B2D).withOpacity(0.07),
                      ),
                    ),
                  ),

                  // ===========================================================
                  // TAP HINT
                  // ===========================================================
                  if (!_pinSet)
                    Positioned(
                      left: 24,
                      right: 24,
                      top: constraints.maxHeight * 0.40,
                      child: IgnorePointer(
                        child: Center(
                          child: Container(
                            padding: const EdgeInsets.symmetric(
                              horizontal: 17,
                              vertical: 11,
                            ),
                            decoration: BoxDecoration(
                              color: const Color(0xFF012E31).withOpacity(0.88),
                              borderRadius: BorderRadius.circular(18),
                              border: Border.all(
                                color: AppTheme.tealMint.withOpacity(0.25),
                              ),
                              boxShadow: [
                                BoxShadow(
                                  color: Colors.black.withOpacity(0.22),
                                  blurRadius: 18,
                                  offset: const Offset(0, 7),
                                ),
                              ],
                            ),
                            child: Row(
                              mainAxisSize: MainAxisSize.min,
                              children: [
                                const Icon(
                                  Icons.touch_app_rounded,
                                  color: AppTheme.tealMint,
                                  size: 18,
                                ),
                                const SizedBox(width: 8),
                                Flexible(
                                  child: Text(
                                    'Tap on map to pin hostel',
                                    textAlign: TextAlign.center,
                                    style: GoogleFonts.dmSans(
                                      color: Colors.white.withOpacity(0.88),
                                      fontSize: 11.5,
                                      fontWeight: FontWeight.w600,
                                    ),
                                  ),
                                ),
                              ],
                            ),
                          ),
                        ),
                      ),
                    ),

                  // ===========================================================
                  // TOP BAR
                  // ===========================================================
                  Positioned(
                    top: safeTop + 10,
                    left: 14,
                    right: 14,
                    child: _buildTopArea(),
                  ),

                  // ===========================================================
                  // STEP BADGE
                  // ===========================================================
                  Positioned(
                    top: safeTop + 67,
                    left: 14,
                    child: _buildStepBadge(),
                  ),

                  // ===========================================================
                  // ATTENDANCE BOUNDARY
                  // ===========================================================
                  Positioned(
                    left: 14,
                    bottom: panelHeight + safeBottom + 18,
                    child: Column(
                      children: [
                        _mapButton(
                          _drawingBoundary
                              ? Icons.check_rounded
                              : Icons.draw_rounded,
                          () {
                            setState(
                              () => _drawingBoundary = !_drawingBoundary,
                            );
                            if (!_drawingBoundary) return;
                            _showMobileMessage(
                              'Tap the outer corners of the hostel building, gate, and entrance.',
                            );
                          },
                        ),
                        if (_drawingBoundary &&
                            _attendanceBoundary.isNotEmpty) ...[
                          const SizedBox(height: 8),
                          _mapButton(
                            Icons.undo_rounded,
                            () => setState(
                              () => _attendanceBoundary.removeLast(),
                            ),
                          ),
                        ],
                      ],
                    ),
                  ),

                  // ===========================================================
                  // ZOOM BUTTONS
                  // ===========================================================
                  Positioned(
                    right: 14,
                    bottom: panelHeight + safeBottom + 18,
                    child: Column(
                      children: [
                        _mapButton(Icons.add_rounded, _zoomIn),
                        const SizedBox(height: 8),
                        _mapButton(Icons.remove_rounded, _zoomOut),
                      ],
                    ),
                  ),

                  // ===========================================================
                  // BOTTOM PANEL
                  // ===========================================================
                  Positioned(
                    left: 0,
                    right: 0,
                    bottom: 0,
                    child: AnimatedBuilder(
                      animation: _panelAnim,
                      builder: (context, child) {
                        return Transform.translate(
                          offset: Offset(0, _panelSlide.value),
                          child: Opacity(
                            opacity: _panelFade.value,
                            child: child,
                          ),
                        );
                      },
                      child: _buildBottomPanel(safeBottom),
                    ),
                  ),
                ],
              );
            },
          ),
        ),
      ),
    );
  }

  // ===========================================================================
  // TOP AREA
  // ===========================================================================

  Widget _buildTopArea() {
    return Column(
      children: [
        Row(
          children: [
            _circleGlassButton(
              icon: Icons.arrow_back_rounded,
              onTap: () {
                Navigator.pop(context);
              },
            ),

            const SizedBox(width: 10),

            Expanded(
              child: Container(
                height: 47,
                decoration: BoxDecoration(
                  color: const Color(0xFF032F32).withOpacity(0.92),
                  borderRadius: BorderRadius.circular(17),
                  border: Border.all(
                    color: AppTheme.tealMint.withOpacity(0.20),
                  ),
                  boxShadow: [
                    BoxShadow(
                      color: Colors.black.withOpacity(0.22),
                      blurRadius: 17,
                      offset: const Offset(0, 6),
                    ),
                  ],
                ),
                child: Row(
                  children: [
                    const SizedBox(width: 13),

                    Icon(
                      Icons.search_rounded,
                      color: AppTheme.tealMint.withOpacity(0.85),
                      size: 19,
                    ),

                    const SizedBox(width: 8),

                    Expanded(
                      child: TextField(
                        controller: _searchCtrl,
                        focusNode: _searchFocus,
                        style: GoogleFonts.dmSans(
                          fontSize: 12.5,
                          color: const Color(0xFF062F32),
                          fontWeight: FontWeight.w700,
                        ),
                        decoration: InputDecoration(
                          filled: true,
                          fillColor: const Color(0xFFF7F9F7),
                          border: OutlineInputBorder(
                            borderRadius: BorderRadius.circular(14),
                            borderSide: BorderSide.none,
                          ),
                          isDense: true,
                          contentPadding: const EdgeInsets.symmetric(
                            horizontal: 12,
                            vertical: 13,
                          ),
                          hintText: 'House, street & sector...',
                          hintStyle: GoogleFonts.dmSans(
                            color: const Color(0xFF728281),
                            fontSize: 12,
                          ),
                        ),
                        onTap: () {
                          if (_suggestions.isNotEmpty) {
                            setState(() {
                              _showSuggestions = true;
                            });
                          }
                        },
                        onChanged: _queueSearch,
                        onSubmitted: (query) {
                          _searchDebounce?.cancel();
                          _search(query);
                        },
                      ),
                    ),

                    if (_searching)
                      Padding(
                        padding: const EdgeInsets.only(right: 13),
                        child: SizedBox(
                          width: 16,
                          height: 16,
                          child: CircularProgressIndicator(
                            strokeWidth: 2,
                            color: AppTheme.tealMint,
                          ),
                        ),
                      )
                    else if (_searchCtrl.text.isNotEmpty)
                      IconButton(
                        splashRadius: 18,
                        onPressed: () {
                          _searchRequest++;
                          _searchCtrl.clear();

                          setState(() {
                            _suggestions = [];
                            _showSuggestions = false;
                          });

                          _searchFocus.requestFocus();
                        },
                        icon: Icon(
                          Icons.close_rounded,
                          size: 17,
                          color: Colors.white.withOpacity(0.55),
                        ),
                      ),
                  ],
                ),
              ),
            ),
          ],
        ),

        // ===============================================================
        // SEARCH RESULTS
        // ===============================================================
        if (_showSuggestions && _suggestions.isNotEmpty) ...[
          const SizedBox(height: 7),

          Align(
            alignment: Alignment.centerRight,
            child: Container(
              width: MediaQuery.sizeOf(context).width - 70,
              constraints: const BoxConstraints(maxHeight: 260),
              decoration: BoxDecoration(
                color: const Color(0xFF032F32).withOpacity(0.97),
                borderRadius: BorderRadius.circular(17),
                border: Border.all(color: AppTheme.tealMint.withOpacity(0.20)),
                boxShadow: [
                  BoxShadow(
                    color: Colors.black.withOpacity(0.25),
                    blurRadius: 20,
                    offset: const Offset(0, 8),
                  ),
                ],
              ),
              child: ListView.separated(
                shrinkWrap: true,
                padding: const EdgeInsets.symmetric(vertical: 5),
                itemCount: _suggestions.length,
                separatorBuilder: (context, index) {
                  return Divider(
                    height: 1,
                    indent: 50,
                    color: Colors.white.withOpacity(0.07),
                  );
                },
                itemBuilder: (context, index) {
                  final suggestion = _suggestions[index];

                  return _buildSuggestionTile(suggestion);
                },
              ),
            ),
          ),
        ],
      ],
    );
  }

  Widget _buildSuggestionTile(Map<String, dynamic> suggestion) {
    final name = suggestion['name'].toString();

    final parts = name.split(',');

    final title = parts.isNotEmpty ? parts.first.trim() : name;

    final subtitle = parts.length > 1 ? parts.sublist(1).join(',').trim() : '';

    return InkWell(
      onTap: () {
        _selectSuggestion(suggestion);
      },
      child: Padding(
        padding: const EdgeInsets.fromLTRB(10, 9, 10, 9),
        child: Row(
          children: [
            Container(
              width: 33,
              height: 33,
              decoration: BoxDecoration(
                color: AppTheme.teal.withOpacity(0.20),
                borderRadius: BorderRadius.circular(10),
              ),
              child: const Icon(
                Icons.location_on_rounded,
                color: AppTheme.tealMint,
                size: 16,
              ),
            ),

            const SizedBox(width: 9),

            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    title,
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis,
                    style: GoogleFonts.dmSans(
                      color: Colors.white,
                      fontSize: 11,
                      fontWeight: FontWeight.w700,
                    ),
                  ),

                  if (subtitle.isNotEmpty) ...[
                    const SizedBox(height: 2),
                    Text(
                      subtitle,
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                      style: GoogleFonts.dmSans(
                        color: Colors.white.withOpacity(0.48),
                        fontSize: 8.5,
                      ),
                    ),
                  ],
                ],
              ),
            ),

            const SizedBox(width: 7),

            const Icon(
              Icons.north_east_rounded,
              color: AppTheme.tealMint,
              size: 14,
            ),
          ],
        ),
      ),
    );
  }

  // ===========================================================================
  // STEP BADGE
  // ===========================================================================

  Widget _buildStepBadge() {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 11, vertical: 7),
      decoration: BoxDecoration(
        color: const Color(0xFF032F32).withOpacity(0.90),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: AppTheme.tealMint.withOpacity(0.20)),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.18),
            blurRadius: 12,
            offset: const Offset(0, 5),
          ),
        ],
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          const Icon(Icons.map_rounded, size: 14, color: AppTheme.tealMint),

          const SizedBox(width: 6),

          Text(
            'Step 4',
            style: GoogleFonts.dmSans(
              color: Colors.white,
              fontSize: 9.5,
              fontWeight: FontWeight.w700,
            ),
          ),
        ],
      ),
    );
  }

  // ===========================================================================
  // MAP MARKER
  // ===========================================================================

  Widget _buildMapMarker() {
    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        Container(
          width: 46,
          height: 46,
          decoration: BoxDecoration(
            gradient: const LinearGradient(
              begin: Alignment.topLeft,
              end: Alignment.bottomRight,
              colors: [Color(0xFF0E8F90), Color(0xFF056669)],
            ),
            shape: BoxShape.circle,
            border: Border.all(color: Colors.white, width: 2.5),
            boxShadow: [
              BoxShadow(
                color: AppTheme.teal.withOpacity(0.42),
                blurRadius: 15,
                offset: const Offset(0, 6),
              ),
            ],
          ),
          child: const Icon(
            Icons.apartment_rounded,
            color: Colors.white,
            size: 22,
          ),
        ),

        Container(
          width: 3,
          height: 10,
          decoration: BoxDecoration(
            color: AppTheme.teal,
            borderRadius: BorderRadius.circular(2),
          ),
        ),

        Container(
          width: 9,
          height: 9,
          decoration: BoxDecoration(
            color: AppTheme.teal.withOpacity(0.40),
            shape: BoxShape.circle,
          ),
        ),
      ],
    );
  }

  // ===========================================================================
  // BOTTOM PANEL
  // ===========================================================================

  Widget _buildBottomPanel(double safeBottom) {
    return Container(
      padding: EdgeInsets.fromLTRB(18, 12, 18, 16 + safeBottom),
      decoration: BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
          colors: [
            const Color(0xFF063F42).withOpacity(0.96),
            const Color(0xFF023336).withOpacity(0.97),
            const Color(0xFF021F22).withOpacity(0.98),
          ],
        ),
        borderRadius: const BorderRadius.vertical(top: Radius.circular(28)),
        border: Border(
          top: BorderSide(
            color: AppTheme.tealMint.withOpacity(0.20),
            width: 0.8,
          ),
        ),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.30),
            blurRadius: 28,
            offset: const Offset(0, -8),
          ),
        ],
      ),
      child: SafeArea(
        top: false,
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Container(
              width: 38,
              height: 4,
              margin: const EdgeInsets.only(bottom: 13),
              decoration: BoxDecoration(
                color: AppTheme.tealMint.withOpacity(0.30),
                borderRadius: BorderRadius.circular(3),
              ),
            ),

            // ---------------------------------------------------------------
            // LOCATION STATUS
            // ---------------------------------------------------------------
            Row(
              children: [
                Container(
                  width: 43,
                  height: 43,
                  decoration: BoxDecoration(
                    color: AppTheme.teal.withOpacity(0.22),
                    borderRadius: BorderRadius.circular(13),
                    border: Border.all(
                      color: AppTheme.tealMint.withOpacity(0.17),
                    ),
                  ),
                  child: const Icon(
                    Icons.location_on_rounded,
                    color: AppTheme.tealMint,
                    size: 21,
                  ),
                ),

                const SizedBox(width: 11),

                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        _attendanceBoundary.length >= 3
                            ? 'Hostel location boundary ready'
                            : _pinSet
                            ? 'Location Pinned'
                            : 'Pin Hostel Location',
                        maxLines: 1,
                        overflow: TextOverflow.ellipsis,
                        style: GoogleFonts.dmSans(
                          color: Colors.white,
                          fontSize: 12.5,
                          fontWeight: FontWeight.w800,
                        ),
                      ),

                      const SizedBox(height: 2),

                      Text(
                        _attendanceBoundary.length >= 3
                            ? '${_attendanceBoundary.length} points cover building, gate & entrance'
                            : _pinSet
                            ? '${_pinnedLocation.latitude.toStringAsFixed(5)}, ${_pinnedLocation.longitude.toStringAsFixed(5)}'
                            : 'Tap the map or search above',
                        maxLines: 1,
                        overflow: TextOverflow.ellipsis,
                        style: GoogleFonts.dmSans(
                          color: Colors.white.withOpacity(0.52),
                          fontSize: 8.8,
                        ),
                      ),
                    ],
                  ),
                ),

                if (_pinSet) ...[
                  const SizedBox(width: 8),
                  Container(
                    padding: const EdgeInsets.symmetric(
                      horizontal: 9,
                      vertical: 5,
                    ),
                    decoration: BoxDecoration(
                      color: const Color(0xFF54C49A).withOpacity(0.16),
                      borderRadius: BorderRadius.circular(10),
                    ),
                    child: Row(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        const Icon(
                          Icons.check_circle_rounded,
                          color: Color(0xFF78D9B5),
                          size: 13,
                        ),
                        const SizedBox(width: 4),
                        Text(
                          'Set',
                          style: GoogleFonts.dmSans(
                            color: const Color(0xFF78D9B5),
                            fontSize: 8.5,
                            fontWeight: FontWeight.w800,
                          ),
                        ),
                      ],
                    ),
                  ),
                ],
              ],
            ),

            const SizedBox(height: 12),

            // ---------------------------------------------------------------
            // HOSTEL SUMMARY
            // ---------------------------------------------------------------
            Container(
              width: double.infinity,
              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
              decoration: BoxDecoration(
                color: const Color(0xFF011F21).withOpacity(0.64),
                borderRadius: BorderRadius.circular(13),
                border: Border.all(color: Colors.white.withOpacity(0.10)),
              ),
              child: Row(
                children: [
                  const Icon(
                    Icons.apartment_rounded,
                    size: 16,
                    color: AppTheme.tealMint,
                  ),

                  const SizedBox(width: 8),

                  Expanded(
                    child: Text(
                      widget.hostelName,
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                      style: GoogleFonts.dmSans(
                        color: Colors.white,
                        fontSize: 10.5,
                        fontWeight: FontWeight.w700,
                      ),
                    ),
                  ),

                  const SizedBox(width: 10),

                  Flexible(
                    child: Text(
                      widget.city,
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                      textAlign: TextAlign.end,
                      style: GoogleFonts.dmSans(
                        color: Colors.white.withOpacity(0.48),
                        fontSize: 9,
                      ),
                    ),
                  ),
                ],
              ),
            ),

            const SizedBox(height: 12),

            // ---------------------------------------------------------------
            // NEXT
            // ---------------------------------------------------------------
            SizedBox(
              width: double.infinity,
              height: 49,
              child: ElevatedButton(
                onPressed: _next,
                style: ElevatedButton.styleFrom(
                  backgroundColor: _pinSet
                      ? AppTheme.teal
                      : const Color(0xFF426568),
                  foregroundColor: Colors.white,
                  elevation: 0,
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(15),
                  ),
                ),
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    Text(
                      'Next: Room Details',
                      style: GoogleFonts.dmSans(
                        fontSize: 13.5,
                        fontWeight: FontWeight.w800,
                      ),
                    ),

                    const SizedBox(width: 8),

                    Container(
                      width: 25,
                      height: 25,
                      decoration: BoxDecoration(
                        color: Colors.white.withOpacity(0.14),
                        shape: BoxShape.circle,
                      ),
                      child: const Icon(Icons.arrow_forward_rounded, size: 14),
                    ),
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  // ===========================================================================
  // BUTTONS
  // ===========================================================================

  Widget _circleGlassButton({
    required IconData icon,
    required VoidCallback onTap,
  }) {
    return InkWell(
      onTap: onTap,
      borderRadius: BorderRadius.circular(50),
      child: Container(
        width: 44,
        height: 44,
        decoration: BoxDecoration(
          color: const Color(0xFF032F32).withOpacity(0.92),
          shape: BoxShape.circle,
          border: Border.all(color: AppTheme.tealMint.withOpacity(0.20)),
          boxShadow: [
            BoxShadow(
              color: Colors.black.withOpacity(0.22),
              blurRadius: 14,
              offset: const Offset(0, 5),
            ),
          ],
        ),
        child: Icon(icon, color: Colors.white, size: 20),
      ),
    );
  }

  Widget _mapButton(IconData icon, VoidCallback onTap) {
    return InkWell(
      onTap: onTap,
      borderRadius: BorderRadius.circular(50),
      child: Container(
        width: 42,
        height: 42,
        decoration: BoxDecoration(
          color: const Color(0xFF032F32).withOpacity(0.94),
          shape: BoxShape.circle,
          border: Border.all(color: AppTheme.tealMint.withOpacity(0.22)),
          boxShadow: [
            BoxShadow(
              color: Colors.black.withOpacity(0.22),
              blurRadius: 12,
              offset: const Offset(0, 5),
            ),
          ],
        ),
        child: Icon(icon, size: 20, color: AppTheme.tealMint),
      ),
    );
  }
}
