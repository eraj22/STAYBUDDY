import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:flutter_map/flutter_map.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:http/http.dart' as http;
import 'package:latlong2/latlong.dart';
import '../theme.dart';
import '../widgets/responsive_shell.dart';
import 'owner_hostel_rooms_screen.dart';

class OwnerHostelMapScreen extends StatefulWidget {
  final String ownerEmail, ownerName, hostelName, hostelType, city, address;
  const OwnerHostelMapScreen({super.key,
    required this.ownerEmail, required this.ownerName,
    required this.hostelName, required this.hostelType,
    required this.city, required this.address});

  @override
  State<OwnerHostelMapScreen> createState() => _OwnerHostelMapScreenState();
}

class _OwnerHostelMapScreenState extends State<OwnerHostelMapScreen>
    with SingleTickerProviderStateMixin {
  final MapController _mapCtrl = MapController();
  final TextEditingController _searchCtrl = TextEditingController();

  // Default: Islamabad
  LatLng _pinnedLocation = const LatLng(33.6844, 73.0479);
  bool _pinSet = false;
  bool _searching = false;
  List<Map<String, dynamic>> _suggestions = [];
  bool _showSuggestions = false;

  late final AnimationController _panelAnim;
  late final Animation<double> _panelSlide;
  late final Animation<double> _panelFade;

  @override
  void initState() {
    super.initState();
    _panelAnim = AnimationController(vsync: this, duration: const Duration(milliseconds: 600));
    _panelSlide = Tween<double>(begin: 80, end: 0).animate(
      CurvedAnimation(parent: _panelAnim, curve: Curves.easeOutCubic));
    _panelFade = CurvedAnimation(parent: _panelAnim, curve: Curves.easeOut);
    _panelAnim.forward();
  }

  @override
  void dispose() {
    _panelAnim.dispose();
    _searchCtrl.dispose();
    super.dispose();
  }

  Future<void> _search(String q) async {
    if (q.trim().isEmpty) { setState(() { _suggestions = []; _showSuggestions = false; }); return; }
    setState(() => _searching = true);
    try {
      final uri = Uri.parse('https://nominatim.openstreetmap.org/search?q=${Uri.encodeComponent(q)}&format=json&limit=5');
      final res = await http.get(uri, headers: {'User-Agent': 'staybuddy_owner'});
      if (!mounted) return;
      final data = jsonDecode(res.body) as List;
      setState(() {
        _suggestions = data.map((e) => {
          'name': e['display_name'] ?? '',
          'lat': double.tryParse(e['lat'].toString()) ?? 0,
          'lng': double.tryParse(e['lon'].toString()) ?? 0,
        }).toList();
        _showSuggestions = true;
        _searching = false;
      });
    } catch (_) {
      if (mounted) setState(() => _searching = false);
    }
  }

  void _selectSuggestion(Map<String, dynamic> s) {
    final loc = LatLng(s['lat'], s['lng']);
    setState(() {
      _pinnedLocation = loc;
      _pinSet = true;
      _showSuggestions = false;
      _searchCtrl.text = (s['name'] as String).split(',').first.trim();
    });
    _mapCtrl.move(loc, 16);
  }

  void _onMapTap(TapPosition _, LatLng latlng) {
    setState(() { _pinnedLocation = latlng; _pinSet = true; _showSuggestions = false; });
    FocusScope.of(context).unfocus();
  }

  void _next() {
    if (!_pinSet) {
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(
        content: Text('Please tap on the map to pin your hostel location.',
          style: GoogleFonts.dmSans(fontSize: 13)),
        backgroundColor: Colors.orange.shade600,
        behavior: SnackBarBehavior.floating,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
        margin: const EdgeInsets.all(16),
      ));
      return;
    }
    Navigator.push(context, MaterialPageRoute(
      builder: (_) => OwnerHostelRoomsScreen(
        ownerEmail: widget.ownerEmail,
        ownerName: widget.ownerName,
        hostelName: widget.hostelName,
        hostelType: widget.hostelType,
        city: widget.city,
        address: widget.address,
        latitude: _pinnedLocation.latitude,
        longitude: _pinnedLocation.longitude,
      ),
    ));
  }

  @override
  Widget build(BuildContext context) {
    return ResponsiveShell(
      child: Scaffold(
        body: GestureDetector(
          onTap: () { FocusScope.of(context).unfocus(); setState(() => _showSuggestions = false); },
          child: Stack(children: [
            // Map
            FlutterMap(
              mapController: _mapCtrl,
              options: MapOptions(
                initialCenter: _pinnedLocation, initialZoom: 14,
                onTap: _onMapTap,
              ),
              children: [
                TileLayer(urlTemplate: 'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
                  userAgentPackageName: 'staybuddy_owner'),
                MarkerLayer(markers: [
                  if (_pinSet) Marker(
                    point: _pinnedLocation, width: 60, height: 70,
                    child: Column(mainAxisSize: MainAxisSize.min, children: [
                      Container(width: 44, height: 44,
                        decoration: BoxDecoration(color: AppTheme.teal, shape: BoxShape.circle,
                          boxShadow: [BoxShadow(color: AppTheme.teal.withOpacity(0.4), blurRadius: 12, offset: const Offset(0, 4))],
                          border: Border.all(color: Colors.white, width: 2.5)),
                        child: const Icon(Icons.business_rounded, color: Colors.white, size: 22)),
                      Container(width: 2, height: 12,
                        decoration: BoxDecoration(color: AppTheme.teal, borderRadius: BorderRadius.circular(1))),
                      Container(width: 8, height: 8,
                        decoration: BoxDecoration(color: AppTheme.teal.withOpacity(0.3), shape: BoxShape.circle)),
                    ]),
                  ),
                ]),
              ],
            ),

            // Tap hint overlay (when no pin yet)
            if (!_pinSet)
              Center(child: IgnorePointer(
                child: Column(mainAxisSize: MainAxisSize.min, children: [
                  Container(
                    padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 12),
                    decoration: BoxDecoration(color: Colors.black.withOpacity(0.55), borderRadius: BorderRadius.circular(20)),
                    child: Row(mainAxisSize: MainAxisSize.min, children: [
                      const Icon(Icons.touch_app_rounded, color: Colors.white, size: 18),
                      const SizedBox(width: 8),
                      Text('Tap on map to pin your hostel', style: GoogleFonts.dmSans(color: Colors.white, fontSize: 13, fontWeight: FontWeight.w500)),
                    ]),
                  ),
                ]),
              )),

            // Top bar: back + step + search
            Positioned(
              top: MediaQuery.of(context).padding.top + 10, left: 14, right: 14,
              child: Column(children: [
                Row(children: [
                  GestureDetector(
                    onTap: () => Navigator.pop(context),
                    child: Container(width: 42, height: 42,
                      decoration: BoxDecoration(color: Colors.white, shape: BoxShape.circle,
                        boxShadow: [BoxShadow(color: Colors.black.withOpacity(0.12), blurRadius: 10)]),
                      child: const Icon(Icons.arrow_back_rounded, size: 20, color: Color(0xFF0A3D3F))),
                  ),
                  const SizedBox(width: 10),
                  Expanded(child: Container(
                    height: 46,
                    decoration: BoxDecoration(color: Colors.white, borderRadius: BorderRadius.circular(23),
                      boxShadow: [BoxShadow(color: Colors.black.withOpacity(0.1), blurRadius: 16, offset: const Offset(0, 4))]),
                    child: Row(children: [
                      const SizedBox(width: 14),
                      Icon(Icons.search_rounded, color: AppTheme.textMuted, size: 20),
                      const SizedBox(width: 8),
                      Expanded(child: TextField(
                        controller: _searchCtrl,
                        style: GoogleFonts.dmSans(fontSize: 13, color: AppTheme.textPrimary),
                        decoration: InputDecoration(
                          border: InputBorder.none,
                          hintText: 'Search location...',
                          hintStyle: GoogleFonts.dmSans(fontSize: 13, color: AppTheme.textMuted)),
                        onChanged: (v) => _search(v),
                      )),
                      if (_searching) Padding(padding: const EdgeInsets.only(right: 12),
                        child: SizedBox(width: 16, height: 16, child: CircularProgressIndicator(strokeWidth: 2, color: AppTheme.teal))),
                    ]),
                  )),
                ]),
                // Suggestions dropdown
                if (_showSuggestions && _suggestions.isNotEmpty) ...[
                  const SizedBox(height: 6),
                  Container(
                    decoration: BoxDecoration(color: Colors.white, borderRadius: BorderRadius.circular(16),
                      boxShadow: [BoxShadow(color: Colors.black.withOpacity(0.12), blurRadius: 20)]),
                    child: ListView.separated(
                      shrinkWrap: true, physics: const NeverScrollableScrollPhysics(),
                      itemCount: _suggestions.length,
                      separatorBuilder: (_, __) => Divider(height: 0.5, color: AppTheme.border),
                      itemBuilder: (_, i) {
                        final s = _suggestions[i];
                        final parts = (s['name'] as String).split(',');
                        return ListTile(
                          dense: true,
                          leading: Container(width: 30, height: 30,
                            decoration: BoxDecoration(color: AppTheme.teal.withOpacity(0.1), shape: BoxShape.circle),
                            child: Icon(Icons.location_on_rounded, size: 15, color: AppTheme.teal)),
                          title: Text(parts.first.trim(), style: GoogleFonts.dmSans(fontSize: 13, fontWeight: FontWeight.w600, color: AppTheme.textPrimary)),
                          subtitle: parts.length > 1 ? Text(parts.sublist(1).join(',').trim(), maxLines: 1, overflow: TextOverflow.ellipsis,
                            style: GoogleFonts.dmSans(fontSize: 11, color: AppTheme.textMuted)) : null,
                          onTap: () => _selectSuggestion(s),
                        );
                      },
                    ),
                  ),
                ],
              ]),
            ),

            // Bottom panel
            Positioned(left: 0, right: 0, bottom: 0,
              child: AnimatedBuilder(
                animation: _panelAnim,
                builder: (_, child) => Transform.translate(
                  offset: Offset(0, _panelSlide.value),
                  child: Opacity(opacity: _panelFade.value, child: child)),
                child: Container(
                  padding: EdgeInsets.fromLTRB(20, 18, 20, MediaQuery.of(context).padding.bottom + 18),
                  decoration: BoxDecoration(color: Colors.white,
                    borderRadius: const BorderRadius.vertical(top: Radius.circular(28)),
                    boxShadow: [BoxShadow(color: Colors.black.withOpacity(0.12), blurRadius: 24, offset: const Offset(0, -4))]),
                  child: Column(mainAxisSize: MainAxisSize.min, children: [
                    Center(child: Container(width: 40, height: 4, margin: const EdgeInsets.only(bottom: 16),
                      decoration: BoxDecoration(color: AppTheme.border, borderRadius: BorderRadius.circular(2)))),

                    Row(children: [
                      Container(width: 42, height: 42,
                        decoration: BoxDecoration(color: AppTheme.teal.withOpacity(0.1), borderRadius: BorderRadius.circular(12)),
                        child: Icon(Icons.location_on_rounded, color: AppTheme.teal, size: 22)),
                      const SizedBox(width: 12),
                      Expanded(child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                        Text(_pinSet ? 'Location Pinned!' : 'Pin Hostel Location',
                          style: GoogleFonts.dmSans(fontSize: 15, fontWeight: FontWeight.w700,
                            color: _pinSet ? AppTheme.teal : AppTheme.textPrimary)),
                        Text(_pinSet
                            ? '${_pinnedLocation.latitude.toStringAsFixed(5)}, ${_pinnedLocation.longitude.toStringAsFixed(5)}'
                            : 'Tap on the map or search above',
                          style: GoogleFonts.dmSans(fontSize: 11, color: AppTheme.textMuted)),
                      ])),
                      if (_pinSet) Container(
                        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
                        decoration: BoxDecoration(color: const Color(0xFFEAF3DE), borderRadius: BorderRadius.circular(10)),
                        child: Row(children: [
                          Icon(Icons.check_circle_rounded, size: 13, color: const Color(0xFF3B6D11)),
                          const SizedBox(width: 4),
                          Text('Set', style: GoogleFonts.dmSans(fontSize: 11, fontWeight: FontWeight.w600, color: const Color(0xFF3B6D11))),
                        ]),
                      ),
                    ]),
                    const SizedBox(height: 14),

                    // Hostel name summary
                    Container(
                      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
                      decoration: BoxDecoration(color: AppTheme.bg, borderRadius: BorderRadius.circular(12),
                        border: Border.all(color: AppTheme.border, width: 0.5)),
                      child: Row(children: [
                        Icon(Icons.business_rounded, size: 16, color: AppTheme.textMuted),
                        const SizedBox(width: 8),
                        Text(widget.hostelName, style: GoogleFonts.dmSans(fontSize: 13, fontWeight: FontWeight.w600, color: AppTheme.textPrimary)),
                        const Spacer(),
                        Text(widget.city, style: GoogleFonts.dmSans(fontSize: 12, color: AppTheme.textMuted)),
                      ]),
                    ),
                    const SizedBox(height: 14),

                    SizedBox(width: double.infinity, height: 50,
                      child: ElevatedButton(
                        onPressed: _next,
                        style: ElevatedButton.styleFrom(
                          backgroundColor: _pinSet ? AppTheme.teal : AppTheme.textMuted,
                          foregroundColor: Colors.white, elevation: 0,
                          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16))),
                        child: Row(mainAxisAlignment: MainAxisAlignment.center, children: [
                          Text('Next: Room Details', style: GoogleFonts.dmSans(fontSize: 15, fontWeight: FontWeight.w600)),
                          const SizedBox(width: 10),
                          Container(width: 26, height: 26,
                            decoration: BoxDecoration(color: Colors.white.withOpacity(0.2), shape: BoxShape.circle),
                            child: const Icon(Icons.arrow_forward_rounded, size: 14)),
                        ]),
                      )),
                  ]),
                ),
              )),

            // Zoom controls
            Positioned(right: 14, bottom: 180,
              child: Column(children: [
                _mapBtn(Icons.add_rounded, () { _mapCtrl.move(_mapCtrl.camera.center, _mapCtrl.camera.zoom + 1); }),
                const SizedBox(height: 8),
                _mapBtn(Icons.remove_rounded, () { _mapCtrl.move(_mapCtrl.camera.center, _mapCtrl.camera.zoom - 1); }),
              ])),

            // Step indicator overlay
            Positioned(top: MediaQuery.of(context).padding.top + 66, left: 14,
              child: Container(
                padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                decoration: BoxDecoration(color: Colors.white,
                  borderRadius: BorderRadius.circular(20),
                  boxShadow: [BoxShadow(color: Colors.black.withOpacity(0.1), blurRadius: 8)]),
                child: Row(mainAxisSize: MainAxisSize.min, children: [
                  Icon(Icons.map_rounded, size: 14, color: AppTheme.teal),
                  const SizedBox(width: 6),
                  Text('Step 4 of 4', style: GoogleFonts.dmSans(fontSize: 11, fontWeight: FontWeight.w600, color: AppTheme.textPrimary)),
                ]),
              )),
          ]),
        ),
      ),
    );
  }

  Widget _mapBtn(IconData icon, VoidCallback onTap) {
    return GestureDetector(
      onTap: onTap,
      child: Container(width: 40, height: 40,
        decoration: BoxDecoration(color: Colors.white, shape: BoxShape.circle,
          border: Border.all(color: AppTheme.border, width: 0.5),
          boxShadow: [BoxShadow(color: Colors.black.withOpacity(0.1), blurRadius: 8)]),
        child: Icon(icon, size: 20, color: AppTheme.textPrimary)),
    );
  }
}