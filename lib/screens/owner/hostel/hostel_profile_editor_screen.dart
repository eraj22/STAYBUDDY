import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:flutter_map/flutter_map.dart';
import 'package:flutter/services.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:image_picker/image_picker.dart';
import 'package:latlong2/latlong.dart';

import '../../../api.dart';
import '../../../theme.dart';
import '../../../widgets/responsive_shell.dart';

class HostelProfileEditorScreen extends StatefulWidget {
  final String hostelName;
  final String initialSection;

  const HostelProfileEditorScreen({
    super.key,
    required this.hostelName,
    this.initialSection = 'details',
  });

  @override
  State<HostelProfileEditorScreen> createState() =>
      _HostelProfileEditorScreenState();
}

class _HostelProfileEditorScreenState extends State<HostelProfileEditorScreen> {
  final _formKey = GlobalKey<FormState>();
  final _detailsKey = GlobalKey();
  final _locationKey = GlobalKey();
  final _photosKey = GlobalKey();
  final _picker = ImagePicker();

  final Map<String, TextEditingController> _c = {
    for (final key in [
      'name',
      'city',
      'area',
      'address',
      'university',
      'description',
      'phone',
      'email',
      'floors',
      'latitude',
      'longitude',
      'rules',
      'mess_charges',
      'meals_per_day',
      'meal_type',
      'meal_timings',
      'menu_items',
      'curfew_time',
      'attendance_type',
      'guest_policy',
      'payment_methods',
      'fine_details',
      'cctv_areas',
      'guard_shifts',
    ])
      key: TextEditingController(),
  };

  int? _hostelId;
  int _rooms = 0;
  int _beds = 0;
  int _available = 0;
  String _type = 'girls';
  bool _loading = true;
  bool _saving = false;
  final Set<String> _facilities = {};
  List<Map<String, dynamic>> _media = [];
  List<Map<String, dynamic>> _roomTypes = [];
  Map<String, dynamic> _policies = {};
  final List<LatLng> _attendanceBoundary = [];
  bool _editingBoundary = false;

  static const _facilityChoices = [
    'WiFi',
    'CCTV',
    'Laundry',
    'Mess',
    'Parking',
    'Security',
    'Generator',
    'Air Conditioning',
    'Attached Bath',
    'Wardrobe',
    'Study Table',
    'Common Areas',
    'Water Filter',
    'Transport',
  ];

  @override
  void initState() {
    super.initState();
    _c['name']!.text = widget.hostelName;
    _load();
  }

  List<dynamic> _list(dynamic value) => value is List ? value : const [];
  Map<String, dynamic> _map(dynamic value) =>
      value is Map ? Map<String, dynamic>.from(value) : {};
  bool _flag(String key) => _policies[key] == true;
  List<String> _split(String text) => text
      .split(RegExp(r'[,\n]'))
      .map((e) => e.trim())
      .where((e) => e.isNotEmpty)
      .toList();
  String _join(dynamic value) =>
      _list(value).map((e) => e.toString()).join(', ');

  Future<void> _load() async {
    try {
      final rows = await Api().getOwnerHostels();
      Map<String, dynamic>? hostel;
      for (final raw in rows.whereType<Map>()) {
        final item = Map<String, dynamic>.from(raw);
        if (item['name']?.toString().trim().toLowerCase() ==
            widget.hostelName.trim().toLowerCase()) {
          hostel = item;
          break;
        }
      }
      hostel ??= rows.whereType<Map>().isEmpty
          ? null
          : Map<String, dynamic>.from(rows.whereType<Map>().first);
      if (hostel == null) throw Exception('No hostel found for this owner.');

      _hostelId = int.tryParse(hostel['id']?.toString() ?? '');
      void text(String key, dynamic value) =>
          _c[key]!.text = value?.toString() ?? '';
      text('name', hostel['name']);
      text('city', hostel['city']);
      text('area', hostel['area']);
      text('address', hostel['address']);
      text('university', hostel['university']);
      text('description', hostel['description']);
      text('phone', hostel['contact_phone']);
      text('email', hostel['contact_email']);
      text('floors', hostel['floors']);
      text('latitude', hostel['latitude']);
      text('longitude', hostel['longitude']);
      _attendanceBoundary
        ..clear()
        ..addAll(
          _list(hostel['attendance_boundary']).whereType<Map>().map(
            (point) => LatLng(
              (point['latitude'] as num).toDouble(),
              (point['longitude'] as num).toDouble(),
            ),
          ),
        );
      _type =
          hostel['hostel_type']?.toString().toLowerCase().contains('boy') ==
              true
          ? 'boys'
          : 'girls';
      _rooms = int.tryParse(hostel['total_rooms']?.toString() ?? '') ?? 0;
      _beds = int.tryParse(hostel['total_capacity']?.toString() ?? '') ?? 0;
      _available =
          int.tryParse(hostel['available_capacity']?.toString() ?? '') ?? 0;

      _facilities
        ..clear()
        ..addAll(
          <dynamic>{
            ..._list(hostel['amenities']),
            ..._list(hostel['facilities']),
          }.map((e) => e.toString()).where((e) => e.isNotEmpty),
        );
      text('rules', _list(hostel['rules']).join('\n'));
      _policies = _map(hostel['mess_info']);
      for (final key in [
        'mess_charges',
        'meals_per_day',
        'meal_type',
        'curfew_time',
        'attendance_type',
        'guest_policy',
        'fine_details',
      ]) {
        text(key, _policies[key]);
      }
      for (final key in [
        'meal_timings',
        'menu_items',
        'payment_methods',
        'cctv_areas',
        'guard_shifts',
      ]) {
        text(key, _join(_policies[key]));
      }
      _media = _list(
        hostel['media'],
      ).whereType<Map>().map((e) => Map<String, dynamic>.from(e)).toList();
      _roomTypes = _list(
        hostel['room_types_available'],
      ).whereType<Map>().map((e) => Map<String, dynamic>.from(e)).toList();

      if (!mounted) return;
      setState(() => _loading = false);
      WidgetsBinding.instance.addPostFrameCallback((_) {
        final key = widget.initialSection == 'photos'
            ? _photosKey
            : widget.initialSection == 'location'
            ? _locationKey
            : _detailsKey;
        if (key.currentContext != null) {
          Scrollable.ensureVisible(
            key.currentContext!,
            duration: const Duration(milliseconds: 450),
            curve: Curves.easeOut,
          );
        }
      });
    } catch (error) {
      if (!mounted) return;
      setState(() => _loading = false);
      _message('Hostel details load nahi ho sakin: $error', error: true);
    }
  }

  Future<void> _pickPhoto() async {
    if (_media.length >= 6) {
      _message('Maximum 6 hostel photos allowed.', error: true);
      return;
    }
    final file = await _picker.pickImage(
      source: ImageSource.gallery,
      imageQuality: 65,
      maxWidth: 1600,
    );
    if (file == null) return;
    final bytes = await file.readAsBytes();
    if (bytes.length > 1500000) {
      _message('Photo 1.5 MB se choti select karein.', error: true);
      return;
    }
    final mime = file.name.toLowerCase().endsWith('.png') ? 'png' : 'jpeg';
    setState(() {
      _media.add({
        'type': 'image',
        'name': file.name,
        'data_url': 'data:image/$mime;base64,${base64Encode(bytes)}',
      });
    });
  }

  Future<void> _save() async {
    FocusScope.of(context).unfocus();
    if (!(_formKey.currentState?.validate() ?? false)) return;
    if (_hostelId == null) {
      return _message('Hostel id missing hai.', error: true);
    }

    final lat = double.tryParse(_c['latitude']!.text.trim());
    final lng = double.tryParse(_c['longitude']!.text.trim());
    if ((lat == null) != (lng == null)) {
      return _message('Latitude aur longitude dono enter karein.', error: true);
    }
    if (_editingBoundary && _attendanceBoundary.length < 3) {
      return _message(
        'Attendance boundary ke liye minimum 3 corners select karein.',
        error: true,
      );
    }
    setState(() => _saving = true);
    try {
      final policies = <String, dynamic>{
        ..._policies,
        for (final key in [
          'mess_charges',
          'meal_type',
          'curfew_time',
          'attendance_type',
          'guest_policy',
          'fine_details',
        ])
          key: _c[key]!.text.trim(),
        'meals_per_day': int.tryParse(_c['meals_per_day']!.text.trim()) ?? 0,
        for (final key in [
          'meal_timings',
          'menu_items',
          'payment_methods',
          'cctv_areas',
          'guard_shifts',
        ])
          key: _split(_c[key]!.text),
      };
      await Api().updateOwnerHostel(
        hostelId: _hostelId!,
        updates: {
          'name': _c['name']!.text.trim(),
          'hostel_type': _type,
          'city': _c['city']!.text.trim(),
          'area': _c['area']!.text.trim(),
          'address': _c['address']!.text.trim(),
          'university': _c['university']!.text.trim(),
          'description': _c['description']!.text.trim(),
          'contact_phone': _c['phone']!.text.trim(),
          'contact_email': _c['email']!.text.trim().toLowerCase(),
          if (_c['floors']!.text.trim().isNotEmpty)
            'floors': int.parse(_c['floors']!.text.trim()),
          if (lat != null) 'latitude': lat,
          if (lng != null) 'longitude': lng,
          'amenities': _facilities.toList()..sort(),
          'facilities': _facilities.toList()..sort(),
          'rules': _split(_c['rules']!.text),
          'mess_info': policies,
          'media': _media,
          'room_types_available': _roomTypes,
        },
      );
      if (_attendanceBoundary.length >= 3) {
        await Api().updateOwnerHostelAttendanceBoundary(
          hostelId: _hostelId!,
          boundary: _attendanceBoundary
              .map(
                (point) => {
                  'latitude': point.latitude,
                  'longitude': point.longitude,
                },
              )
              .toList(growable: false),
        );
      }
      if (!mounted) return;
      Navigator.pop(context, {
        'hostelName': _c['name']!.text.trim(),
        'hostelType': _type,
        'city': _c['city']!.text.trim(),
        'address': _c['address']!.text.trim(),
        'phone': _c['phone']!.text.trim(),
        'email': _c['email']!.text.trim(),
      });
    } catch (error) {
      if (!mounted) return;
      setState(() => _saving = false);
      _message('Changes save nahi ho sakin: $error', error: true);
    }
  }

  @override
  void dispose() {
    for (final controller in _c.values) {
      controller.dispose();
    }
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return ResponsiveShell(
      child: Scaffold(
        backgroundColor: AppTheme.bg,
        body: Column(
          children: [
            _header(),
            Expanded(
              child: _loading
                  ? const Center(
                      child: CircularProgressIndicator(color: AppTheme.teal),
                    )
                  : Form(
                      key: _formKey,
                      child: ListView(
                        keyboardDismissBehavior:
                            ScrollViewKeyboardDismissBehavior.onDrag,
                        padding: const EdgeInsets.fromLTRB(18, 6, 18, 32),
                        children: [
                          _section(
                            _detailsKey,
                            'Basic Information',
                            'Hostel identity and public profile',
                          ),
                          _field(
                            'name',
                            'Hostel Name',
                            Icons.apartment_rounded,
                            required: true,
                          ),
                          _typeSelector(),
                          _field(
                            'university',
                            'Nearby University / Institute',
                            Icons.school_outlined,
                          ),
                          _field(
                            'description',
                            'Hostel Description',
                            Icons.notes_rounded,
                            lines: 5,
                          ),
                          _section(
                            _locationKey,
                            'Location & Map',
                            'Full address and exact database coordinates',
                          ),
                          _field(
                            'city',
                            'City',
                            Icons.location_city_rounded,
                            required: true,
                          ),
                          _field('area', 'Area / Sector', Icons.map_outlined),
                          _field(
                            'address',
                            'Full Address',
                            Icons.location_on_outlined,
                            required: true,
                            lines: 3,
                          ),
                          Row(
                            children: [
                              Expanded(
                                child: _field(
                                  'latitude',
                                  'Latitude',
                                  Icons.my_location_rounded,
                                  keyboard:
                                      const TextInputType.numberWithOptions(
                                        decimal: true,
                                        signed: true,
                                      ),
                                ),
                              ),
                              const SizedBox(width: 10),
                              Expanded(
                                child: _field(
                                  'longitude',
                                  'Longitude',
                                  Icons.explore_outlined,
                                  keyboard:
                                      const TextInputType.numberWithOptions(
                                        decimal: true,
                                        signed: true,
                                      ),
                                ),
                              ),
                            ],
                          ),
                          _locationMap(),
                          _section(
                            null,
                            'Contact Details',
                            'Public hostel phone and email',
                          ),
                          _field(
                            'phone',
                            'Hostel Phone',
                            Icons.phone_outlined,
                            required: true,
                            phone: true,
                            keyboard: TextInputType.phone,
                          ),
                          _field(
                            'email',
                            'Hostel Email',
                            Icons.email_outlined,
                            required: true,
                            keyboard: TextInputType.emailAddress,
                            email: true,
                          ),
                          _section(
                            null,
                            'Accommodation',
                            'Rooms and beds remain consistent with Rooms & Beds',
                          ),
                          _summary(),
                          _field(
                            'floors',
                            'Total Floors',
                            Icons.layers_outlined,
                            keyboard: TextInputType.number,
                            digits: true,
                          ),
                          if (_roomTypes.isNotEmpty) _roomTypeSummary(),
                          _section(
                            null,
                            'Amenities & Facilities',
                            'Tap to add or remove facilities',
                          ),
                          _facilityGrid(),
                          _section(
                            null,
                            'Mess & Food',
                            'Meals, charges, timings and menu',
                          ),
                          _toggle('Mess Included', 'mess_included'),
                          _toggle('Extra Mess Charges', 'extra_mess_charges'),
                          _field(
                            'mess_charges',
                            'Mess Charges (PKR)',
                            Icons.payments_outlined,
                            keyboard: TextInputType.number,
                            digits: true,
                          ),
                          _field(
                            'meals_per_day',
                            'Meals Per Day',
                            Icons.restaurant_outlined,
                            keyboard: TextInputType.number,
                            digits: true,
                          ),
                          _field(
                            'meal_type',
                            'Meal Type',
                            Icons.dinner_dining_outlined,
                          ),
                          _field(
                            'meal_timings',
                            'Meal Timings',
                            Icons.schedule_outlined,
                          ),
                          _field(
                            'menu_items',
                            'Menu Items',
                            Icons.menu_book_outlined,
                            lines: 2,
                          ),
                          _section(
                            null,
                            'Safety & Operations',
                            'Attendance, security, warden and gate controls',
                          ),
                          for (final item in const [
                            ('Attendance System', 'attendance_system'),
                            ('Entry Card System', 'entry_card_system'),
                            ('CCTV Installed', 'cctv_installed'),
                            ('Warden Onsite', 'warden_onsite'),
                            ('Hostel Curfew', 'has_curfew'),
                            ('Late Entry Allowed', 'late_entry_allowed'),
                            (
                              'Notify Parent on Absence',
                              'parent_absence_notification',
                            ),
                          ])
                            _toggle(item.$1, item.$2),
                          _field(
                            'cctv_areas',
                            'CCTV Areas',
                            Icons.videocam_outlined,
                          ),
                          _field(
                            'guard_shifts',
                            'Guard Shifts',
                            Icons.security_outlined,
                          ),
                          _field(
                            'curfew_time',
                            'Curfew Time',
                            Icons.schedule_rounded,
                          ),
                          _field(
                            'attendance_type',
                            'Attendance Type',
                            Icons.how_to_reg_outlined,
                          ),
                          _section(
                            null,
                            'Rules & Payment Policies',
                            'Guest, payment, fine and resident rules',
                          ),
                          _field(
                            'guest_policy',
                            'Guest Policy',
                            Icons.groups_outlined,
                          ),
                          _field(
                            'payment_methods',
                            'Payment Methods',
                            Icons.account_balance_wallet_outlined,
                          ),
                          _toggle('Fine Policy', 'fine_policy'),
                          _field(
                            'fine_details',
                            'Fine Details',
                            Icons.receipt_long_outlined,
                            lines: 2,
                          ),
                          _field(
                            'rules',
                            'Hostel Rules (one per line)',
                            Icons.rule_rounded,
                            lines: 6,
                          ),
                          _section(
                            _photosKey,
                            'Hostel Photos',
                            'Gallery images saved with this hostel',
                          ),
                          _photos(),
                        ],
                      ),
                    ),
            ),
            _bottomBar(),
          ],
        ),
      ),
    );
  }

  Widget _header() => Container(
    padding: EdgeInsets.fromLTRB(
      12,
      MediaQuery.paddingOf(context).top + 8,
      18,
      20,
    ),
    decoration: const BoxDecoration(
      gradient: LinearGradient(colors: [Color(0xFF073F42), Color(0xFF0A797B)]),
      borderRadius: BorderRadius.vertical(bottom: Radius.circular(30)),
    ),
    child: Row(
      children: [
        IconButton(
          onPressed: () => Navigator.pop(context),
          icon: const Icon(Icons.arrow_back_rounded, color: Colors.white),
        ),
        const SizedBox(width: 4),
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                'Edit Hostel Profile',
                style: GoogleFonts.playfairDisplay(
                  color: Colors.white,
                  fontSize: 25,
                  fontWeight: FontWeight.w700,
                ),
              ),
              Text(
                'View and update every saved hostel detail',
                style: GoogleFonts.dmSans(
                  color: Colors.white70,
                  fontSize: 10.5,
                ),
              ),
            ],
          ),
        ),
      ],
    ),
  );

  Widget _section(Key? key, String title, String subtitle) => Padding(
    key: key,
    padding: const EdgeInsets.only(top: 22, bottom: 11),
    child: Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Container(
          width: 4,
          height: 34,
          decoration: BoxDecoration(
            color: AppTheme.teal,
            borderRadius: BorderRadius.circular(4),
          ),
        ),
        const SizedBox(width: 10),
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                title,
                style: GoogleFonts.dmSans(
                  fontSize: 15,
                  fontWeight: FontWeight.w800,
                  color: AppTheme.textPrimary,
                ),
              ),
              Text(
                subtitle,
                style: GoogleFonts.dmSans(
                  fontSize: 9.5,
                  color: AppTheme.textMuted,
                ),
              ),
            ],
          ),
        ),
      ],
    ),
  );

  Widget _field(
    String key,
    String label,
    IconData icon, {
    bool required = false,
    bool email = false,
    bool phone = false,
    bool digits = false,
    int lines = 1,
    TextInputType? keyboard,
  }) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 10),
      child: TextFormField(
        controller: _c[key],
        maxLines: lines,
        keyboardType: keyboard,
        inputFormatters: digits
            ? [FilteringTextInputFormatter.digitsOnly]
            : phone
            ? [FilteringTextInputFormatter.allow(RegExp(r'[0-9+\-\s]'))]
            : null,
        validator: (value) {
          final text = value?.trim() ?? '';
          if (required && text.isEmpty) return '$label is required';
          if (email &&
              text.isNotEmpty &&
              !RegExp(
                r"^[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)+$",
              ).hasMatch(text)) {
            return 'Enter a valid email address';
          }
          if (phone && text.isNotEmpty) {
            final compact = text.replaceAll(RegExp(r'[\s-]'), '');
            if (!RegExp(r'^(?:03\d{9}|\+923\d{9})$').hasMatch(compact)) {
              return 'Use 03XXXXXXXXX or +923XXXXXXXXX';
            }
          }
          return null;
        },
        decoration: InputDecoration(
          labelText: label,
          prefixIcon: Icon(icon, color: AppTheme.teal),
          filled: true,
          fillColor: Colors.white,
          enabledBorder: OutlineInputBorder(
            borderRadius: BorderRadius.circular(16),
            borderSide: BorderSide(color: AppTheme.border),
          ),
          focusedBorder: OutlineInputBorder(
            borderRadius: BorderRadius.circular(16),
            borderSide: const BorderSide(color: AppTheme.teal, width: 1.5),
          ),
          border: OutlineInputBorder(borderRadius: BorderRadius.circular(16)),
        ),
      ),
    );
  }

  Widget _typeSelector() => Padding(
    padding: const EdgeInsets.only(bottom: 10),
    child: SegmentedButton<String>(
      segments: const [
        ButtonSegment(value: 'girls', label: Text('Girls Hostel')),
        ButtonSegment(value: 'boys', label: Text('Boys Hostel')),
      ],
      selected: {_type},
      onSelectionChanged: (value) => setState(() => _type = value.first),
    ),
  );

  Widget _locationMap() {
    final latitude = double.tryParse(_c['latitude']!.text.trim());
    final longitude = double.tryParse(_c['longitude']!.text.trim());
    final center = _attendanceBoundary.isNotEmpty
        ? _attendanceBoundary.first
        : LatLng(latitude ?? 33.6844, longitude ?? 73.0479);
    final pin = latitude != null && longitude != null
        ? LatLng(latitude, longitude)
        : center;

    return Container(
      margin: const EdgeInsets.only(bottom: 12),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(18),
        border: Border.all(color: AppTheme.border),
      ),
      clipBehavior: Clip.antiAlias,
      child: Column(
        children: [
          SizedBox(
            height: 285,
            child: FlutterMap(
              options: MapOptions(
                initialCenter: center,
                initialZoom: 17,
                onTap: (_, point) {
                  setState(() {
                    _c['latitude']!.text = point.latitude.toStringAsFixed(8);
                    _c['longitude']!.text = point.longitude.toStringAsFixed(8);
                    if (_editingBoundary) _attendanceBoundary.add(point);
                  });
                },
              ),
              children: [
                TileLayer(
                  urlTemplate: 'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
                  userAgentPackageName: 'com.staybuddy.app',
                ),
                if (_attendanceBoundary.length >= 3)
                  PolygonLayer(
                    polygons: [
                      Polygon(
                        points: _attendanceBoundary,
                        color: AppTheme.teal.withOpacity(.20),
                        borderColor: const Color(0xFFFFA640),
                        borderStrokeWidth: 3,
                      ),
                    ],
                  ),
                MarkerLayer(
                  markers: [
                    Marker(
                      point: pin,
                      width: 46,
                      height: 46,
                      child: Container(
                        decoration: BoxDecoration(
                          color: AppTheme.teal,
                          shape: BoxShape.circle,
                          border: Border.all(color: Colors.white, width: 3),
                        ),
                        child: const Icon(
                          Icons.apartment_rounded,
                          color: Colors.white,
                          size: 23,
                        ),
                      ),
                    ),
                    ..._attendanceBoundary.map(
                      (point) => Marker(
                        point: point,
                        width: 16,
                        height: 16,
                        child: Container(
                          decoration: BoxDecoration(
                            color: const Color(0xFFFFA640),
                            shape: BoxShape.circle,
                            border: Border.all(color: Colors.white, width: 2),
                          ),
                        ),
                      ),
                    ),
                  ],
                ),
              ],
            ),
          ),
          Padding(
            padding: const EdgeInsets.all(12),
            child: Column(
              children: [
                Row(
                  children: [
                    Expanded(
                      child: Text(
                        _editingBoundary
                            ? 'Tap around building, gate and entrance (minimum 3 corners).'
                            : '${_attendanceBoundary.length} boundary corners saved',
                        style: GoogleFonts.dmSans(
                          fontSize: 10.5,
                          color: AppTheme.textMuted,
                        ),
                      ),
                    ),
                    Switch.adaptive(
                      value: _editingBoundary,
                      activeColor: AppTheme.teal,
                      onChanged: (value) =>
                          setState(() => _editingBoundary = value),
                    ),
                  ],
                ),
                if (_editingBoundary)
                  Row(
                    children: [
                      Expanded(
                        child: OutlinedButton.icon(
                          onPressed: _attendanceBoundary.isEmpty
                              ? null
                              : () => setState(
                                  () => _attendanceBoundary.removeLast(),
                                ),
                          icon: const Icon(Icons.undo_rounded),
                          label: const Text('Undo corner'),
                        ),
                      ),
                      const SizedBox(width: 8),
                      Expanded(
                        child: OutlinedButton.icon(
                          onPressed: () =>
                              setState(() => _attendanceBoundary.clear()),
                          icon: const Icon(Icons.refresh_rounded),
                          label: const Text('Redraw'),
                        ),
                      ),
                    ],
                  ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _summary() => Container(
    margin: const EdgeInsets.only(bottom: 10),
    padding: const EdgeInsets.all(15),
    decoration: BoxDecoration(
      color: const Color(0xFFE9F4F2),
      borderRadius: BorderRadius.circular(17),
      border: Border.all(color: AppTheme.teal.withOpacity(.2)),
    ),
    child: Row(
      mainAxisAlignment: MainAxisAlignment.spaceAround,
      children: [
        _metric('$_rooms', 'Rooms'),
        _metric('$_beds', 'Beds'),
        _metric('$_available', 'Available'),
      ],
    ),
  );

  Widget _metric(String value, String label) => Column(
    children: [
      Text(
        value,
        style: GoogleFonts.dmSans(
          color: AppTheme.teal,
          fontSize: 20,
          fontWeight: FontWeight.w800,
        ),
      ),
      Text(
        label,
        style: GoogleFonts.dmSans(color: AppTheme.textMuted, fontSize: 9.5),
      ),
    ],
  );

  Widget _roomTypeSummary() => Container(
    width: double.infinity,
    margin: const EdgeInsets.only(bottom: 8),
    padding: const EdgeInsets.all(13),
    decoration: BoxDecoration(
      color: Colors.white,
      borderRadius: BorderRadius.circular(16),
      border: Border.all(color: AppTheme.border),
    ),
    child: Wrap(
      spacing: 6,
      runSpacing: 6,
      children: _roomTypes.map((room) {
        final label = room['type'] ?? room['room_type'] ?? 'Room type';
        final rent = room['rent'] ?? room['price'] ?? room['monthly_rent'];
        return Chip(
          label: Text(rent == null ? '$label' : '$label · PKR $rent'),
        );
      }).toList(),
    ),
  );

  Widget _facilityGrid() => Wrap(
    spacing: 7,
    runSpacing: 7,
    children: _facilityChoices.map((name) {
      final selected = _facilities.contains(name);
      return FilterChip(
        selected: selected,
        label: Text(name),
        avatar: Icon(
          selected
              ? Icons.check_circle_rounded
              : Icons.add_circle_outline_rounded,
        ),
        onSelected: (value) => setState(
          () => value ? _facilities.add(name) : _facilities.remove(name),
        ),
        selectedColor: AppTheme.teal.withOpacity(.16),
      );
    }).toList(),
  );

  Widget _toggle(String title, String key) => Container(
    margin: const EdgeInsets.only(bottom: 9),
    decoration: BoxDecoration(
      color: Colors.white,
      borderRadius: BorderRadius.circular(15),
      border: Border.all(color: AppTheme.border),
    ),
    child: SwitchListTile.adaptive(
      title: Text(
        title,
        style: GoogleFonts.dmSans(fontWeight: FontWeight.w700, fontSize: 12.5),
      ),
      value: _flag(key),
      activeColor: AppTheme.teal,
      onChanged: (value) => setState(() => _policies[key] = value),
    ),
  );

  Widget _photos() => Column(
    children: [
      if (_media.isEmpty)
        Container(
          width: double.infinity,
          padding: const EdgeInsets.all(20),
          decoration: BoxDecoration(
            color: Colors.white,
            borderRadius: BorderRadius.circular(16),
            border: Border.all(color: AppTheme.border),
          ),
          child: const Text(
            'No hostel photos saved yet.',
            textAlign: TextAlign.center,
          ),
        ),
      ..._media.asMap().entries.map(
        (entry) => Card(
          margin: const EdgeInsets.only(bottom: 8),
          child: ListTile(
            leading: _preview(entry.value),
            title: Text(entry.value['name']?.toString() ?? 'Hostel photo'),
            subtitle: const Text('Saved with hostel profile'),
            trailing: IconButton(
              icon: const Icon(Icons.delete_outline_rounded, color: Colors.red),
              onPressed: () => setState(() => _media.removeAt(entry.key)),
            ),
          ),
        ),
      ),
      const SizedBox(height: 4),
      SizedBox(
        width: double.infinity,
        child: OutlinedButton.icon(
          onPressed: _pickPhoto,
          icon: const Icon(Icons.add_photo_alternate_outlined),
          label: const Text('Add Hostel Photo'),
        ),
      ),
    ],
  );

  Widget _preview(Map<String, dynamic> item) {
    final data = item['data_url']?.toString() ?? '';
    if (data.contains(',')) {
      try {
        return ClipRRect(
          borderRadius: BorderRadius.circular(9),
          child: Image.memory(
            base64Decode(data.split(',').last),
            width: 48,
            height: 48,
            fit: BoxFit.cover,
          ),
        );
      } catch (_) {}
    }
    return const CircleAvatar(child: Icon(Icons.image_outlined));
  }

  Widget _bottomBar() => SafeArea(
    top: false,
    child: Container(
      padding: const EdgeInsets.fromLTRB(18, 10, 18, 12),
      decoration: BoxDecoration(
        color: AppTheme.bg,
        border: Border(top: BorderSide(color: AppTheme.border)),
      ),
      child: SizedBox(
        height: 52,
        width: double.infinity,
        child: ElevatedButton.icon(
          onPressed: _loading || _saving ? null : _save,
          icon: _saving
              ? const SizedBox(
                  width: 20,
                  height: 20,
                  child: CircularProgressIndicator(
                    strokeWidth: 2,
                    color: Colors.white,
                  ),
                )
              : const Icon(Icons.cloud_done_outlined),
          label: Text(_saving ? 'Saving...' : 'Save Changes to Database'),
          style: ElevatedButton.styleFrom(
            backgroundColor: AppTheme.teal,
            foregroundColor: Colors.white,
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(16),
            ),
          ),
        ),
      ),
    ),
  );

  void _message(String text, {bool error = false}) {
    ScaffoldMessenger.of(context)
      ..hideCurrentSnackBar()
      ..showSnackBar(
        SnackBar(
          content: Text(text),
          backgroundColor: error ? Colors.red.shade700 : AppTheme.teal,
        ),
      );
  }
}
