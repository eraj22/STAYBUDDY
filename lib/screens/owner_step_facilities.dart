import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import '../theme.dart';
import '../widgets/responsive_shell.dart';
import '../widgets/video_background.dart';
import 'owner_steps_mess_safety_rules.dart';

class OwnerStepFacilities extends StatefulWidget {
  final String ownerEmail, ownerName, hostelName, hostelType, city, address;
  final double latitude, longitude;
  final int totalFloors, totalRooms, totalCapacity;

  const OwnerStepFacilities({
    super.key,
    required this.ownerEmail, required this.ownerName,
    required this.hostelName, required this.hostelType,
    required this.city, required this.address,
    required this.latitude, required this.longitude,
    required this.totalFloors, required this.totalRooms,
    required this.totalCapacity,
  });

  @override
  State<OwnerStepFacilities> createState() => _OwnerStepFacilitiesState();
}

class _OwnerStepFacilitiesState extends State<OwnerStepFacilities>
    with SingleTickerProviderStateMixin {

  // Facilities organised by category
  final _cats = <String, List<Map<String, dynamic>>>{
    'Internet & Tech': [
      {'label': 'WiFi',         'icon': Icons.wifi_rounded,              'sel': true},
      {'label': 'Cable TV',     'icon': Icons.tv_rounded,                'sel': false},
    ],
    'Comfort': [
      {'label': 'AC',           'icon': Icons.ac_unit_rounded,           'sel': false},
      {'label': 'Heater',       'icon': Icons.thermostat_rounded,        'sel': false},
      {'label': 'Attached Bath','icon': Icons.bathroom_rounded,          'sel': false},
      {'label': 'Cupboard',     'icon': Icons.inventory_2_rounded,       'sel': false},
    ],
    'Utilities': [
      {'label': 'Hot Water',    'icon': Icons.water_drop_rounded,        'sel': true},
      {'label': 'Generator',    'icon': Icons.bolt_rounded,              'sel': false},
      {'label': 'Water Filter', 'icon': Icons.water_rounded,             'sel': false},
      {'label': 'Laundry',      'icon': Icons.local_laundry_service_rounded, 'sel': false},
    ],
    'Common Areas': [
      {'label': 'Study Room',   'icon': Icons.menu_book_rounded,         'sel': false},
      {'label': 'Common Room',  'icon': Icons.weekend_rounded,           'sel': false},
      {'label': 'Cafeteria',    'icon': Icons.restaurant_rounded,        'sel': false},
      {'label': 'Library',      'icon': Icons.local_library_rounded,     'sel': false},
    ],
    'Safety': [
      {'label': 'CCTV',         'icon': Icons.videocam_rounded,          'sel': false},
      {'label': 'Security Guard','icon': Icons.security_rounded,         'sel': false},
      {'label': 'Prayer Room',  'icon': Icons.mosque_rounded,            'sel': false},
    ],
    'Transport': [
      {'label': 'Parking',      'icon': Icons.local_parking_rounded,     'sel': false},
      {'label': 'Bus Stop Nearby','icon': Icons.directions_bus_rounded,  'sel': false},
    ],
  };

  final _customCtrl = TextEditingController();
  final _customList = <String>[];

  late final AnimationController _anim;
  late final Animation<double> _fade, _slide;

  @override
  void initState() {
    super.initState();
    _anim = AnimationController(vsync: this,
        duration: const Duration(milliseconds: 800));
    _fade  = CurvedAnimation(parent: _anim,
        curve: const Interval(0.15, 1.0, curve: Curves.easeOut));
    _slide = Tween<double>(begin: 60, end: 0).animate(CurvedAnimation(
        parent: _anim,
        curve: const Interval(0.15, 1.0, curve: Curves.easeOutCubic)));
    _anim.forward();
  }

  @override
  void dispose() {
    _anim.dispose();
    _customCtrl.dispose();
    super.dispose();
  }

  int get _selectedCount => _cats.values
      .expand((items) => items)
      .where((f) => f['sel'] == true)
      .length + _customList.length;

  Color _catColor(String cat) {
    switch (cat) {
      case 'Internet & Tech': return AppTheme.teal;
      case 'Comfort':         return const Color(0xFF7B3FC4);
      case 'Utilities':       return const Color(0xFFB85C00);
      case 'Common Areas':    return const Color(0xFF185FA5);
      case 'Safety':          return Colors.red.shade600;
      case 'Transport':       return Colors.indigo.shade600;
      default:                return AppTheme.teal;
    }
  }

  @override
  Widget build(BuildContext context) {
    return ResponsiveShell(
      child: Scaffold(
        body: VideoBackground(
          assetPath: 'assets/images/background.mp4',
          overlayOpacity: 0.30,
          cardMargin: const EdgeInsets.all(14),
          cardRadius: 32,
          child: SafeArea(
            child: Column(crossAxisAlignment: CrossAxisAlignment.start,
                children: [
              Padding(
                padding: const EdgeInsets.fromLTRB(20, 14, 20, 0),
                child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                  Row(children: [
                    GestureDetector(
                      onTap: () => Navigator.pop(context),
                      child: Container(
                        width: 38, height: 38,
                        decoration: BoxDecoration(
                          color: Colors.white.withOpacity(0.15),
                          shape: BoxShape.circle,
                          border: Border.all(
                              color: Colors.white.withOpacity(0.3))),
                        child: const Icon(Icons.arrow_back_rounded,
                            color: Colors.white, size: 18)),
                    ),
                    const Spacer(),
                    Container(
                      padding: const EdgeInsets.symmetric(
                          horizontal: 12, vertical: 5),
                      decoration: BoxDecoration(
                        color: Colors.white.withOpacity(0.12),
                        borderRadius: BorderRadius.circular(20),
                        border: Border.all(
                            color: Colors.white.withOpacity(0.2))),
                      child: Text('Step 5 of 9  ·  $_selectedCount selected',
                          style: GoogleFonts.dmSans(
                              color: Colors.white, fontSize: 12,
                              fontWeight: FontWeight.w600)),
                    ),
                  ]),
                  const SizedBox(height: 16),
                  Row(children: List.generate(9, (i) => Expanded(
                    child: Container(
                      height: 4,
                      margin: EdgeInsets.only(right: i < 8 ? 4 : 0),
                      decoration: BoxDecoration(
                        color: i < 5
                            ? AppTheme.tealMint
                            : Colors.white.withOpacity(0.25),
                        borderRadius: BorderRadius.circular(2)),
                    ),
                  ))),
                  const SizedBox(height: 14),
                  Text('Facilities & Amenities',
                      style: GoogleFonts.playfairDisplay(
                          fontSize: 30, fontWeight: FontWeight.w700,
                          color: Colors.white)),
                  const SizedBox(height: 4),
                  Text('What does your hostel offer?',
                      style: GoogleFonts.dmSans(
                          color: Colors.white.withOpacity(0.6),
                          fontSize: 13)),
                ]),
              ),
              Expanded(
                child: AnimatedBuilder(
                  animation: _anim,
                  builder: (_, child) => Transform.translate(
                    offset: Offset(0, _slide.value),
                    child: Opacity(opacity: _fade.value, child: child),
                  ),
                  child: DraggableScrollableSheet(
                    initialChildSize: 0.85,
                    minChildSize: 0.45,
                    maxChildSize: 0.96,
                    expand: true,
                    builder: (ctx, scrollCtrl) => _buildSheet(scrollController: scrollCtrl),
                  ),
                ),
              ),
            ]),
          ),
        ),
      ),
    );
  }

  Widget _buildSheet({ScrollController? scrollController}) {
    return Container(
      width: double.infinity,
      decoration: const BoxDecoration(
        color: AppTheme.bg,
        borderRadius: BorderRadius.vertical(top: Radius.circular(32)),
      ),
      child: Column(children: [
        const SizedBox(height: 12),
        Center(child: Container(
          width: 40, height: 4,
          decoration: BoxDecoration(
              color: AppTheme.border,
              borderRadius: BorderRadius.circular(2)))),
        Expanded(
          child: SingleChildScrollView(
            padding: const EdgeInsets.fromLTRB(20, 14, 20, 32),
            child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
              // Category groups
              ..._cats.entries.map((e) {
                final catColor = _catColor(e.key);
                return Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                  // Category label
                  Padding(
                    padding: const EdgeInsets.only(bottom: 8),
                    child: Row(children: [
                      Container(width: 6, height: 6,
                        decoration: BoxDecoration(
                            color: catColor, shape: BoxShape.circle)),
                      const SizedBox(width: 6),
                      Text(e.key.toUpperCase(),
                          style: GoogleFonts.dmSans(
                              fontSize: 10, fontWeight: FontWeight.w700,
                              color: catColor, letterSpacing: 0.8)),
                    ]),
                  ),
                  // Chips
                  Wrap(spacing: 8, runSpacing: 8,
                    children: e.value.asMap().entries.map((fe) {
                      final f = fe.value;
                      final sel = f['sel'] as bool;
                      return GestureDetector(
                        onTap: () => setState(
                            () => f['sel'] = !sel),
                        child: AnimatedContainer(
                          duration: const Duration(milliseconds: 160),
                          padding: const EdgeInsets.symmetric(
                              horizontal: 13, vertical: 8),
                          decoration: BoxDecoration(
                            color: sel
                                ? catColor
                                : AppTheme.card,
                            borderRadius:
                                BorderRadius.circular(12),
                            border: Border.all(
                                color: sel
                                    ? catColor
                                    : AppTheme.border,
                                width: 0.5),
                          ),
                          child: Row(
                              mainAxisSize: MainAxisSize.min,
                              children: [
                            Icon(f['icon'] as IconData,
                                size: 14,
                                color: sel
                                    ? Colors.white
                                    : AppTheme.textMuted),
                            const SizedBox(width: 6),
                            Text(f['label'] as String,
                                style: GoogleFonts.dmSans(
                                    fontSize: 12,
                                    fontWeight: FontWeight.w500,
                                    color: sel
                                        ? Colors.white
                                        : AppTheme.textPrimary)),
                          ]),
                        ),
                      );
                    }).toList(),
                  ),
                  const SizedBox(height: 16),
                ]);
              }),

              // Custom add
              if (_customList.isNotEmpty) ...[
                Padding(
                  padding: const EdgeInsets.only(bottom: 8),
                  child: Row(children: [
                    Container(width: 6, height: 6,
                      decoration: const BoxDecoration(
                          color: AppTheme.tealMint,
                          shape: BoxShape.circle)),
                    const SizedBox(width: 6),
                    Text('ADDITIONAL',
                        style: GoogleFonts.dmSans(
                            fontSize: 10, fontWeight: FontWeight.w700,
                            color: AppTheme.tealMint,
                            letterSpacing: 0.8)),
                  ]),
                ),
                Wrap(spacing: 8, runSpacing: 8,
                  children: _customList.map((name) => Chip(
                    label: Text(name,
                        style: GoogleFonts.dmSans(fontSize: 12)),
                    backgroundColor:
                        AppTheme.teal.withOpacity(0.08),
                    side: BorderSide(
                        color: AppTheme.teal.withOpacity(0.3)),
                    deleteIconColor: AppTheme.teal,
                    onDeleted: () =>
                        setState(() => _customList.remove(name)),
                  )).toList(),
                ),
                const SizedBox(height: 12),
              ],

              Row(children: [
                Expanded(child: _field(_customCtrl,
                    'Add facility not listed above',
                    Icons.add_rounded)),
                const SizedBox(width: 8),
                GestureDetector(
                  onTap: () {
                    final v = _customCtrl.text.trim();
                    if (v.isNotEmpty) {
                      setState(() {
                        _customList.add(v);
                        _customCtrl.clear();
                      });
                    }
                  },
                  child: Container(
                    width: 48, height: 48,
                    decoration: BoxDecoration(
                        color: AppTheme.teal,
                        borderRadius: BorderRadius.circular(14)),
                    child: const Icon(Icons.add_rounded,
                        color: Colors.white, size: 22)),
                ),
              ]),
              const SizedBox(height: 20),

              SizedBox(
                width: double.infinity, height: 52,
                child: ElevatedButton(
                  onPressed: () => Navigator.push(
                      context, MaterialPageRoute(
                    builder: (_) => OwnerStepMess(
                      ownerEmail: widget.ownerEmail,
                      ownerName: widget.ownerName,
                      hostelName: widget.hostelName,
                      hostelType: widget.hostelType,
                      city: widget.city, address: widget.address,
                      latitude: widget.latitude,
                      longitude: widget.longitude,
                    ),
                  )),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: AppTheme.teal,
                    foregroundColor: Colors.white, elevation: 0,
                    shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(16))),
                  child: Row(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                    Text('Next  →  Mess & Food',
                        style: GoogleFonts.dmSans(
                            fontSize: 15,
                            fontWeight: FontWeight.w600)),
                    const SizedBox(width: 10),
                    Container(
                      width: 26, height: 26,
                      decoration: BoxDecoration(
                          color: Colors.white.withOpacity(0.2),
                          shape: BoxShape.circle),
                      child: const Icon(
                          Icons.restaurant_rounded, size: 14)),
                  ]),
                ),
              ),
            ]),
          ),
        ),
      ]),
    );
  }

  Widget _field(TextEditingController ctrl, String hint, IconData icon) {
    return Container(
      decoration: BoxDecoration(
        color: AppTheme.card,
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: AppTheme.border, width: 0.5)),
      child: TextField(
        controller: ctrl,
        style: GoogleFonts.dmSans(
            fontSize: 14, color: AppTheme.textPrimary),
        decoration: InputDecoration(
          hintText: hint,
          hintStyle: GoogleFonts.dmSans(
              color: AppTheme.textMuted, fontSize: 13),
          prefixIcon: Icon(icon, size: 18, color: AppTheme.textMuted),
          border: InputBorder.none,
          contentPadding: const EdgeInsets.symmetric(
              horizontal: 16, vertical: 14)),
      ),
    );
  }
}