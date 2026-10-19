import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:google_fonts/google_fonts.dart';

import '../../../theme.dart';
import '../../../api.dart';
import '../../../widgets/responsive_shell.dart';
import '../../../widgets/video_background.dart';

class AddRoomScreen extends StatefulWidget {
  final String hostelName;

  // Kept only for hot-reload compatibility with the previously running
  // widget. The restored room form uses its built-in floor list below.
  final List<String>? availableFloors;
  final String? initialFloor;

  const AddRoomScreen({
    super.key,
    required this.hostelName,
    this.availableFloors,
    this.initialFloor,
  });

  @override
  State<AddRoomScreen> createState() => _AddRoomScreenState();
}

class _AddRoomScreenState extends State<AddRoomScreen>
    with SingleTickerProviderStateMixin {
  final _roomNoCtrl = TextEditingController();
  final _rentCtrl = TextEditingController();
  final _notesCtrl = TextEditingController();

  final _roomNoFocus = FocusNode();
  final _rentFocus = FocusNode();
  final _notesFocus = FocusNode();

  int _currentStep = 0;
  bool _saving = false;

  String _selectedFloor = 'Ground Floor';

  String _selectedType = '2 Seater';

  int _totalBeds = 2;

  late final List<String> _floors;

  final List<String> _roomTypes = [
    '1 Seater',
    '2 Seater',
    '3 Seater',
    '4 Seater',
    '5 Seater',
    '6 Seater',
  ];

  final Map<String, bool> _amenities = {
    'WiFi': true,
    'Attached Bath': true,
    'Wardrobe': true,
    'Study Table': true,
    'Air Conditioning': false,
    'Balcony': false,
  };

  static const int _totalSteps = 1;

  late final AnimationController _anim;
  late final Animation<double> _fade;
  late final Animation<double> _slide;

  // ===========================================================================
  // INIT
  // ===========================================================================

  @override
  void initState() {
    super.initState();

    _anim = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 700),
    );

    _fade = CurvedAnimation(parent: _anim, curve: Curves.easeOut);

    _slide = Tween<double>(
      begin: 22,
      end: 0,
    ).animate(CurvedAnimation(parent: _anim, curve: Curves.easeOutCubic));

    _anim.forward();

    final suppliedFloors =
        widget.availableFloors
            ?.map((floor) => floor.trim())
            .where((floor) => floor.isNotEmpty)
            .toSet()
            .toList() ??
        <String>[];
    _floors = suppliedFloors.isEmpty ? ['Ground Floor'] : suppliedFloors;
    _selectedFloor =
        widget.initialFloor != null && _floors.contains(widget.initialFloor)
        ? widget.initialFloor!
        : _floors.first;
  }

  @override
  void dispose() {
    _anim.dispose();

    _roomNoCtrl.dispose();
    _rentCtrl.dispose();
    _notesCtrl.dispose();

    _roomNoFocus.dispose();
    _rentFocus.dispose();
    _notesFocus.dispose();

    super.dispose();
  }

  // ===========================================================================
  // VALIDATION
  // ===========================================================================

  bool _validateBasicInfo() {
    final roomNo = _roomNoCtrl.text.trim();

    final rent = int.tryParse(_rentCtrl.text.trim());

    if (roomNo.isEmpty) {
      _showError('Please enter a room number.');
      return false;
    }

    if (rent == null || rent <= 0) {
      _showError('Please enter a valid monthly rent.');
      return false;
    }

    return true;
  }

  // ===========================================================================
  // STEPS
  // ===========================================================================

  void _nextStep() {
    FocusScope.of(context).unfocus();

    if (_currentStep == 0 && !_validateBasicInfo()) {
      return;
    }

    if (_currentStep == _totalSteps - 1) {
      _createRoom();
      return;
    }

    setState(() {
      _currentStep++;
    });
  }

  void _previousStep() {
    FocusScope.of(context).unfocus();

    if (_currentStep == 0) {
      Navigator.pop(context);
      return;
    }

    setState(() {
      _currentStep--;
    });
  }

  // ===========================================================================
  // CREATE ROOM
  // ===========================================================================

  Future<void> _createRoom() async {
    if (_saving) return;

    FocusScope.of(context).unfocus();

    setState(() {
      _saving = true;
    });

    try {
      final dashboard = await Api().getOwnerDashboard();
      final hostels = dashboard['hostels'];

      if (hostels is! List || hostels.isEmpty) {
        throw Exception('No hostel found');
      }

      Map<String, dynamic>? currentHostel;

      for (final item in hostels) {
        if (item is Map) {
          final data = Map<String, dynamic>.from(item);

          if (data['hostel_name']?.toString().trim().toLowerCase() ==
              widget.hostelName.trim().toLowerCase()) {
            currentHostel = data;
            break;
          }
        }
      }

      currentHostel ??= Map<String, dynamic>.from(hostels.first as Map);

      final hostelId = int.tryParse(
        currentHostel['hostel_id']?.toString() ?? '',
      );

      if (hostelId == null) {
        throw Exception('Invalid hostel id');
      }

      final response = await Api().addRoom(
        hostelId: hostelId,
        roomNumber: _roomNoCtrl.text.trim(),
        floorLabel: _selectedFloor,
        capacity: _totalBeds,
        roomType: _selectedType,
        monthlyRent: int.tryParse(_rentCtrl.text.trim()) ?? 0,
        bedNumbers: List.generate(_totalBeds, (index) => '${index + 1}'),
        amenities: _amenities.entries
            .where((entry) => entry.value)
            .map((entry) => entry.key)
            .toList(),
        notes: _notesCtrl.text,
      );

      if (!mounted) return;

      final created = Map<String, dynamic>.from(response['room'] as Map);

      final room = <String, dynamic>{
        'id': created['id'],
        'hostelId': created['hostel_id'],
        'roomNo': created['room_number']?.toString() ?? '',
        'floor': _selectedFloor,
        'type': created['room_type']?.toString() ?? _selectedType,
        'totalBeds': int.tryParse(created['capacity']?.toString() ?? '0') ?? 0,
        'occupiedBeds': 0,
        'rent': int.tryParse(_rentCtrl.text.trim()) ?? 0,
        'status': 'Available',
        'notes': _notesCtrl.text.trim(),
      };

      setState(() {
        _saving = false;
      });

      Navigator.pop(context, room);
    } catch (e) {
      if (!mounted) return;

      setState(() {
        _saving = false;
      });

      ScaffoldMessenger.of(context)
        ..hideCurrentSnackBar()
        ..showSnackBar(SnackBar(content: Text('Could not create room: $e')));
    }
  }
  // ===========================================================================
  // BUILD
  // ===========================================================================

  @override
  Widget build(BuildContext context) {
    return ResponsiveShell(
      child: Scaffold(
        resizeToAvoidBottomInset: true,
        backgroundColor: Colors.transparent,

        body: VideoBackground(
          assetPath: 'assets/images/background.mp4',

          overlayOpacity: 0.48,

          cardMargin: const EdgeInsets.all(14),

          cardRadius: 32,

          child: SafeArea(
            child: Column(
              children: [
                _buildHeader(),

                const SizedBox(height: 10),

                Expanded(
                  child: AnimatedBuilder(
                    animation: _anim,
                    builder: (context, child) {
                      return Transform.translate(
                        offset: Offset(0, _slide.value),
                        child: Opacity(opacity: _fade.value, child: child),
                      );
                    },
                    child: _buildCombinedStep(),
                  ),
                ),

                _buildBottomBar(),
              ],
            ),
          ),
        ),
      ),
    );
  }

  // ===========================================================================
  // HEADER
  // ===========================================================================

  Widget _buildHeader() {
    return Padding(
      padding: const EdgeInsets.fromLTRB(16, 12, 16, 0),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              _circleButton(
                icon: Icons.arrow_back_rounded,
                onTap: _previousStep,
              ),

              const Spacer(),

              Container(
                padding: const EdgeInsets.symmetric(
                  horizontal: 11,
                  vertical: 6,
                ),
                decoration: BoxDecoration(
                  color: const Color(0xFF063F42).withOpacity(0.72),
                  borderRadius: BorderRadius.circular(18),
                  border: Border.all(
                    color: AppTheme.tealMint.withOpacity(0.20),
                  ),
                ),
                child: Text(
                  '${_currentStep + 1} of $_totalSteps',
                  style: GoogleFonts.dmSans(
                    color: Colors.white,
                    fontSize: 10.5,
                    fontWeight: FontWeight.w700,
                  ),
                ),
              ),
            ],
          ),

          const SizedBox(height: 18),

          _buildProgress(),

          const SizedBox(height: 17),

          Text(
            _stepTitle,
            style: GoogleFonts.playfairDisplay(
              fontSize: 28,
              height: 1,
              fontWeight: FontWeight.w700,
              color: Colors.white,
            ),
          ),

          const SizedBox(height: 7),

          Text(
            _stepSubtitle,
            style: GoogleFonts.dmSans(
              fontSize: 11.5,
              color: Colors.white.withOpacity(0.62),
            ),
          ),
        ],
      ),
    );
  }

  Widget _circleButton({required IconData icon, required VoidCallback onTap}) {
    return Material(
      color: Colors.transparent,
      child: InkWell(
        onTap: onTap,
        customBorder: const CircleBorder(),
        child: Container(
          width: 42,
          height: 42,
          decoration: BoxDecoration(
            color: const Color(0xFF063F42).withOpacity(0.60),
            shape: BoxShape.circle,
            border: Border.all(color: Colors.white.withOpacity(0.20)),
          ),
          child: Icon(icon, color: Colors.white, size: 20),
        ),
      ),
    );
  }

  Widget _buildProgress() {
    return Row(
      children: List.generate(_totalSteps, (index) {
        final active = index <= _currentStep;

        return Expanded(
          child: AnimatedContainer(
            duration: const Duration(milliseconds: 220),
            height: index == _currentStep ? 5 : 4,
            margin: EdgeInsets.only(right: index < _totalSteps - 1 ? 7 : 0),
            decoration: BoxDecoration(
              color: active
                  ? AppTheme.tealMint
                  : Colors.white.withOpacity(0.20),
              borderRadius: BorderRadius.circular(8),
            ),
          ),
        );
      }),
    );
  }

  String get _stepTitle {
    return 'Add Room';
  }

  String get _stepSubtitle {
    return 'Floor, room, beds, rent and AC';
  }

  // ===========================================================================
  // STEP 1
  // ===========================================================================

  /// One complete inventory form. An owner should not have to visit separate
  /// pages just to define a room's floor, beds, price and facilities.
  Widget _buildCombinedStep() {
    return _stepScroll(
      children: [
        _introCard(
          icon: Icons.add_home_work_rounded,
          title: 'Room inventory',
          subtitle: 'Add one room with its exact floor, beds, rent and AC.',
        ),
        const SizedBox(height: 20),
        _sectionHeading('Room details'),
        const SizedBox(height: 12),
        _field(
          controller: _roomNoCtrl,
          focusNode: _roomNoFocus,
          label: 'Room Number',
          hint: 'e.g. 101',
          icon: Icons.meeting_room_outlined,
          textInputAction: TextInputAction.next,
          onSubmitted: (_) => FocusScope.of(context).requestFocus(_rentFocus),
        ),
        const SizedBox(height: 13),
        _dropdown(
          label: 'Floor',
          value: _selectedFloor,
          icon: Icons.layers_outlined,
          items: _floors,
          onChanged: (value) => setState(() => _selectedFloor = value!),
        ),
        Align(
          alignment: Alignment.centerLeft,
          child: TextButton.icon(
            onPressed: _addFloor,
            icon: const Icon(Icons.add_rounded, size: 17),
            label: const Text('Add a new floor'),
            style: TextButton.styleFrom(foregroundColor: AppTheme.tealMint),
          ),
        ),
        _field(
          controller: _rentCtrl,
          focusNode: _rentFocus,
          label: 'Monthly Rent (PKR)',
          hint: 'e.g. 23000',
          icon: Icons.payments_outlined,
          keyboardType: TextInputType.number,
          textInputAction: TextInputAction.next,
          formatters: [FilteringTextInputFormatter.digitsOnly],
          onSubmitted: (_) => FocusScope.of(context).requestFocus(_notesFocus),
        ),
        const SizedBox(height: 22),
        _sectionHeading('Seater & beds'),
        const SizedBox(height: 12),
        Wrap(
          spacing: 8,
          runSpacing: 8,
          children: _roomTypes.map((type) {
            final selected = type == _selectedType;
            return ChoiceChip(
              label: Text(type),
              selected: selected,
              onSelected: (_) {
                final beds = int.parse(type.split(' ').first);
                setState(() {
                  _selectedType = type;
                  _totalBeds = beds;
                });
              },
              selectedColor: AppTheme.teal,
              backgroundColor: Colors.black.withOpacity(0.22),
              labelStyle: GoogleFonts.dmSans(
                color: selected ? Colors.white : Colors.white.withOpacity(0.72),
                fontWeight: FontWeight.w700,
                fontSize: 10,
              ),
              side: BorderSide(
                color: selected
                    ? AppTheme.tealMint
                    : Colors.white.withOpacity(0.12),
              ),
            );
          }).toList(),
        ),
        const SizedBox(height: 13),
        _bedCounter(),
        const SizedBox(height: 22),
        _sectionHeading('Room amenities'),
        const SizedBox(height: 12),
        _buildAmenities(),
        const SizedBox(height: 18),
        _multilineField(
          controller: _notesCtrl,
          focusNode: _notesFocus,
          label: 'Room Notes',
          hint: 'Optional notes about this room...',
          icon: Icons.notes_rounded,
        ),
        const SizedBox(height: 24),
      ],
    );
  }

  Future<void> _addFloor() async {
    final controller = TextEditingController();
    final label = await showDialog<String>(
      context: context,
      builder: (dialogContext) => AlertDialog(
        title: const Text('Add floor'),
        content: TextField(
          controller: controller,
          autofocus: true,
          decoration: const InputDecoration(
            hintText: 'e.g. 4th Floor or Basement',
          ),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(dialogContext),
            child: const Text('Cancel'),
          ),
          FilledButton(
            onPressed: () =>
                Navigator.pop(dialogContext, controller.text.trim()),
            child: const Text('Add'),
          ),
        ],
      ),
    );
    controller.dispose();
    if (label == null || label.isEmpty) return;
    if (_floors.any((floor) => floor.toLowerCase() == label.toLowerCase())) {
      _showError('This floor already exists.');
      return;
    }
    setState(() {
      _floors.add(label);
      _selectedFloor = label;
    });
  }

  Widget _buildBasicStep() {
    return _stepScroll(
      children: [
        _introCard(
          icon: Icons.add_home_work_rounded,
          title: 'Create a new room',
          subtitle:
              'Add a room to ${widget.hostelName} and configure its rent and location.',
        ),

        const SizedBox(height: 20),

        _sectionHeading('Basic Information'),

        const SizedBox(height: 12),

        _field(
          controller: _roomNoCtrl,
          focusNode: _roomNoFocus,
          label: 'Room Number',
          hint: 'e.g. 101',
          icon: Icons.meeting_room_outlined,
          textInputAction: TextInputAction.next,
          onSubmitted: (_) {
            FocusScope.of(context).requestFocus(_rentFocus);
          },
        ),

        const SizedBox(height: 13),

        _dropdown(
          label: 'Floor',
          value: _selectedFloor,
          icon: Icons.layers_outlined,
          items: _floors,
          onChanged: (value) {
            if (value == null) {
              return;
            }

            setState(() {
              _selectedFloor = value;
            });
          },
        ),

        const SizedBox(height: 13),

        _field(
          controller: _rentCtrl,
          focusNode: _rentFocus,
          label: 'Monthly Rent',
          hint: 'e.g. 18000',
          icon: Icons.payments_outlined,
          keyboardType: TextInputType.number,
          textInputAction: TextInputAction.next,
          formatters: [FilteringTextInputFormatter.digitsOnly],
          onSubmitted: (_) {
            FocusScope.of(context).requestFocus(_notesFocus);
          },
        ),

        const SizedBox(height: 13),

        _multilineField(
          controller: _notesCtrl,
          focusNode: _notesFocus,
          label: 'Room Notes',
          hint: 'Optional notes about the room...',
          icon: Icons.notes_rounded,
        ),

        const SizedBox(height: 20),
      ],
    );
  }

  // ===========================================================================
  // STEP 2
  // ===========================================================================

  Widget _buildConfigurationStep() {
    return _stepScroll(
      children: [
        _introCard(
          icon: Icons.tune_rounded,
          title: 'Room setup',
          subtitle:
              'Set sharing type, number of beds and available facilities.',
        ),

        const SizedBox(height: 20),

        _sectionHeading('Room Type'),

        const SizedBox(height: 12),

        ..._roomTypes.map(
          (type) => Padding(
            padding: const EdgeInsets.only(bottom: 9),
            child: _roomTypeTile(type),
          ),
        ),

        const SizedBox(height: 14),

        _sectionHeading('Number of Beds'),

        const SizedBox(height: 12),

        _bedCounter(),

        const SizedBox(height: 22),

        _sectionHeading('Room Amenities'),

        const SizedBox(height: 12),

        _buildAmenities(),

        const SizedBox(height: 20),
      ],
    );
  }

  Widget _roomTypeTile(String type) {
    final selected = _selectedType == type;

    return Material(
      color: Colors.transparent,
      child: InkWell(
        borderRadius: BorderRadius.circular(17),
        onTap: () {
          setState(() {
            _selectedType = type;

            switch (type) {
              case 'Single Room':
                _totalBeds = 1;
                break;

              case 'Double Sharing':
                _totalBeds = 2;
                break;

              case 'Triple Sharing':
                _totalBeds = 3;
                break;

              default:
                _totalBeds = 4;
            }
          });
        },
        child: AnimatedContainer(
          duration: const Duration(milliseconds: 180),
          padding: const EdgeInsets.all(13),
          decoration: BoxDecoration(
            color: selected
                ? AppTheme.teal.withOpacity(0.20)
                : Colors.black.withOpacity(0.25),
            borderRadius: BorderRadius.circular(17),
            border: Border.all(
              color: selected
                  ? AppTheme.tealMint.withOpacity(0.55)
                  : Colors.white.withOpacity(0.10),
              width: selected ? 1.15 : 0.8,
            ),
          ),
          child: Row(
            children: [
              Container(
                width: 42,
                height: 42,
                decoration: BoxDecoration(
                  color: selected
                      ? AppTheme.teal.withOpacity(0.28)
                      : Colors.white.withOpacity(0.07),
                  borderRadius: BorderRadius.circular(12),
                ),
                child: Icon(
                  _typeIcon(type),
                  size: 20,
                  color: selected
                      ? AppTheme.tealMint
                      : Colors.white.withOpacity(0.52),
                ),
              ),

              const SizedBox(width: 12),

              Expanded(
                child: Text(
                  type,
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                  style: GoogleFonts.dmSans(
                    fontSize: 11.5,
                    fontWeight: FontWeight.w700,
                    color: Colors.white,
                  ),
                ),
              ),

              const SizedBox(width: 8),

              AnimatedContainer(
                duration: const Duration(milliseconds: 180),
                width: 23,
                height: 23,
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  color: selected ? AppTheme.teal : Colors.transparent,
                  border: Border.all(
                    color: selected
                        ? AppTheme.tealMint
                        : Colors.white.withOpacity(0.22),
                    width: 1.4,
                  ),
                ),
                child: selected
                    ? const Icon(
                        Icons.check_rounded,
                        size: 14,
                        color: Colors.white,
                      )
                    : null,
              ),
            ],
          ),
        ),
      ),
    );
  }

  IconData _typeIcon(String type) {
    switch (type) {
      case 'Single Room':
        return Icons.single_bed_rounded;

      case 'Double Sharing':
        return Icons.bed_rounded;

      case 'Triple Sharing':
        return Icons.bedroom_parent_outlined;

      default:
        return Icons.groups_rounded;
    }
  }

  // ===========================================================================
  // BED COUNTER
  // ===========================================================================

  Widget _bedCounter() {
    return _glassCard(
      child: LayoutBuilder(
        builder: (context, constraints) {
          return Row(
            children: [
              Container(
                width: 43,
                height: 43,
                decoration: BoxDecoration(
                  color: AppTheme.teal.withOpacity(0.20),
                  borderRadius: BorderRadius.circular(13),
                ),
                child: const Icon(
                  Icons.bed_rounded,
                  color: AppTheme.tealMint,
                  size: 20,
                ),
              ),

              const SizedBox(width: 11),

              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'Total Beds',
                      style: GoogleFonts.dmSans(
                        fontSize: 11.5,
                        fontWeight: FontWeight.w700,
                        color: Colors.white,
                      ),
                    ),

                    const SizedBox(height: 2),

                    Text(
                      'Maximum 6 beds',
                      style: GoogleFonts.dmSans(
                        fontSize: 8.5,
                        color: Colors.white.withOpacity(0.43),
                      ),
                    ),
                  ],
                ),
              ),

              const SizedBox(width: 6),

              _counterButton(
                icon: Icons.remove_rounded,
                enabled: _totalBeds > 1,
                onTap: () {
                  setState(() {
                    if (_totalBeds > 1) {
                      _totalBeds--;
                    }
                  });
                },
              ),

              Container(
                constraints: const BoxConstraints(minWidth: 36),
                padding: const EdgeInsets.symmetric(horizontal: 6),
                alignment: Alignment.center,
                child: Text(
                  '$_totalBeds',
                  style: GoogleFonts.dmSans(
                    fontSize: 15,
                    fontWeight: FontWeight.w800,
                    color: AppTheme.tealMint,
                  ),
                ),
              ),

              _counterButton(
                icon: Icons.add_rounded,
                enabled: _totalBeds < 6,
                onTap: () {
                  setState(() {
                    if (_totalBeds < 6) {
                      _totalBeds++;
                    }
                  });
                },
              ),
            ],
          );
        },
      ),
    );
  }

  Widget _counterButton({
    required IconData icon,
    required bool enabled,
    required VoidCallback onTap,
  }) {
    return GestureDetector(
      onTap: enabled ? onTap : null,
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 160),
        width: 36,
        height: 36,
        decoration: BoxDecoration(
          color: enabled ? AppTheme.teal : Colors.white.withOpacity(0.07),
          borderRadius: BorderRadius.circular(11),
        ),
        child: Icon(
          icon,
          size: 18,
          color: enabled ? Colors.white : Colors.white.withOpacity(0.25),
        ),
      ),
    );
  }

  // ===========================================================================
  // AMENITIES
  // ===========================================================================

  Widget _buildAmenities() {
    return LayoutBuilder(
      builder: (context, constraints) {
        final columns = constraints.maxWidth < 320 ? 1 : 2;

        const spacing = 9.0;

        final itemWidth = columns == 1
            ? constraints.maxWidth
            : (constraints.maxWidth - spacing) / 2;

        return Wrap(
          spacing: spacing,
          runSpacing: spacing,
          children: _amenities.entries.map((entry) {
            final enabled = entry.value;

            return SizedBox(
              width: itemWidth,
              child: Material(
                color: Colors.transparent,
                child: InkWell(
                  borderRadius: BorderRadius.circular(16),
                  onTap: () {
                    setState(() {
                      _amenities[entry.key] = !enabled;
                    });
                  },
                  child: AnimatedContainer(
                    duration: const Duration(milliseconds: 180),
                    padding: const EdgeInsets.symmetric(
                      horizontal: 11,
                      vertical: 12,
                    ),
                    decoration: BoxDecoration(
                      color: enabled
                          ? AppTheme.teal.withOpacity(0.18)
                          : Colors.black.withOpacity(0.24),
                      borderRadius: BorderRadius.circular(16),
                      border: Border.all(
                        color: enabled
                            ? AppTheme.tealMint.withOpacity(0.40)
                            : Colors.white.withOpacity(0.09),
                      ),
                    ),
                    child: Row(
                      children: [
                        Icon(
                          _amenityIcon(entry.key),
                          size: 17,
                          color: enabled
                              ? AppTheme.tealMint
                              : Colors.white.withOpacity(0.45),
                        ),

                        const SizedBox(width: 8),

                        Expanded(
                          child: Text(
                            entry.key,
                            maxLines: 2,
                            overflow: TextOverflow.ellipsis,
                            style: GoogleFonts.dmSans(
                              fontSize: 9.5,
                              height: 1.2,
                              fontWeight: FontWeight.w600,
                              color: enabled
                                  ? Colors.white
                                  : Colors.white.withOpacity(0.55),
                            ),
                          ),
                        ),

                        const SizedBox(width: 4),

                        Icon(
                          enabled
                              ? Icons.check_circle_rounded
                              : Icons.circle_outlined,
                          size: 16,
                          color: enabled
                              ? AppTheme.tealMint
                              : Colors.white.withOpacity(0.30),
                        ),
                      ],
                    ),
                  ),
                ),
              ),
            );
          }).toList(),
        );
      },
    );
  }

  IconData _amenityIcon(String amenity) {
    switch (amenity) {
      case 'WiFi':
        return Icons.wifi_rounded;

      case 'Attached Bath':
        return Icons.bathtub_outlined;

      case 'Wardrobe':
        return Icons.door_sliding_outlined;

      case 'Study Table':
        return Icons.desk_outlined;

      case 'Air Conditioning':
        return Icons.ac_unit_rounded;

      case 'Balcony':
        return Icons.balcony_outlined;

      default:
        return Icons.check_rounded;
    }
  }

  // ===========================================================================
  // STEP 3 REVIEW
  // ===========================================================================

  Widget _buildReviewStep() {
    final enabledAmenities = _amenities.entries
        .where((entry) => entry.value)
        .map((entry) => entry.key)
        .toList();

    return _stepScroll(
      children: [
        _introCard(
          icon: Icons.fact_check_rounded,
          title: 'Review room',
          subtitle:
              'Confirm the room information before adding it to the hostel.',
        ),

        const SizedBox(height: 20),

        _reviewSection(
          title: 'Room Details',
          icon: Icons.meeting_room_outlined,
          children: [
            _reviewRow(
              'Room',
              _roomNoCtrl.text.trim().isEmpty ? '-' : _roomNoCtrl.text.trim(),
            ),

            _reviewRow('Floor', _selectedFloor),

            _reviewRow('Type', _selectedType),

            _reviewRow('Beds', '$_totalBeds'),

            _reviewRow(
              'Rent',
              'PKR ${_formatAmount(int.tryParse(_rentCtrl.text.trim()) ?? 0)}',
            ),
          ],
        ),

        const SizedBox(height: 13),

        _reviewSection(
          title: 'Amenities',
          icon: Icons.check_circle_outline_rounded,
          children: [
            if (enabledAmenities.isEmpty)
              Text(
                'No amenities selected',
                style: GoogleFonts.dmSans(
                  fontSize: 10,
                  color: Colors.white.withOpacity(0.46),
                ),
              )
            else
              Wrap(
                spacing: 7,
                runSpacing: 7,
                children: enabledAmenities.map((item) {
                  return Container(
                    padding: const EdgeInsets.symmetric(
                      horizontal: 9,
                      vertical: 5,
                    ),
                    decoration: BoxDecoration(
                      color: AppTheme.teal.withOpacity(0.18),
                      borderRadius: BorderRadius.circular(18),
                      border: Border.all(
                        color: AppTheme.tealMint.withOpacity(0.20),
                      ),
                    ),
                    child: Text(
                      item,
                      style: GoogleFonts.dmSans(
                        fontSize: 8.5,
                        fontWeight: FontWeight.w600,
                        color: AppTheme.tealMint,
                      ),
                    ),
                  );
                }).toList(),
              ),
          ],
        ),

        if (_notesCtrl.text.trim().isNotEmpty) ...[
          const SizedBox(height: 13),

          _reviewSection(
            title: 'Notes',
            icon: Icons.notes_rounded,
            children: [
              Text(
                _notesCtrl.text.trim(),
                style: GoogleFonts.dmSans(
                  fontSize: 10,
                  height: 1.5,
                  color: Colors.white.withOpacity(0.72),
                ),
              ),
            ],
          ),
        ],

        const SizedBox(height: 14),

        Container(
          padding: const EdgeInsets.all(13),
          decoration: BoxDecoration(
            color: AppTheme.teal.withOpacity(0.16),
            borderRadius: BorderRadius.circular(16),
            border: Border.all(color: AppTheme.tealMint.withOpacity(0.19)),
          ),
          child: Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Icon(
                Icons.check_circle_outline_rounded,
                color: AppTheme.tealMint,
                size: 18,
              ),

              const SizedBox(width: 9),

              Expanded(
                child: Text(
                  'Everything looks ready. Tap Create Room to add this room to ${widget.hostelName}.',
                  style: GoogleFonts.dmSans(
                    fontSize: 9.5,
                    height: 1.4,
                    color: Colors.white.withOpacity(0.70),
                  ),
                ),
              ),
            ],
          ),
        ),

        const SizedBox(height: 20),
      ],
    );
  }

  // ===========================================================================
  // REVIEW SECTION
  // ===========================================================================

  Widget _reviewSection({
    required String title,
    required IconData icon,
    required List<Widget> children,
  }) {
    return _glassCard(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Container(
                width: 39,
                height: 39,
                decoration: BoxDecoration(
                  color: AppTheme.teal.withOpacity(0.19),
                  borderRadius: BorderRadius.circular(12),
                ),
                child: Icon(icon, color: AppTheme.tealMint, size: 19),
              ),

              const SizedBox(width: 11),

              Expanded(
                child: Text(
                  title,
                  style: GoogleFonts.dmSans(
                    fontSize: 12,
                    fontWeight: FontWeight.w800,
                    color: Colors.white,
                  ),
                ),
              ),
            ],
          ),

          const SizedBox(height: 13),

          ...children,
        ],
      ),
    );
  }

  Widget _reviewRow(String label, String value) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 6),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          SizedBox(
            width: 72,
            child: Text(
              label,
              style: GoogleFonts.dmSans(
                fontSize: 9,
                color: Colors.white.withOpacity(0.44),
              ),
            ),
          ),

          const SizedBox(width: 10),

          Expanded(
            child: Text(
              value,
              textAlign: TextAlign.right,
              maxLines: 3,
              overflow: TextOverflow.ellipsis,
              style: GoogleFonts.dmSans(
                fontSize: 10.5,
                fontWeight: FontWeight.w700,
                color: Colors.white,
              ),
            ),
          ),
        ],
      ),
    );
  }

  // ===========================================================================
  // SCROLL
  // ===========================================================================

  Widget _stepScroll({required List<Widget> children}) {
    final keyboardHeight = MediaQuery.viewInsetsOf(context).bottom;

    return SingleChildScrollView(
      keyboardDismissBehavior: ScrollViewKeyboardDismissBehavior.onDrag,
      padding: EdgeInsets.fromLTRB(16, 12, 16, 24 + keyboardHeight),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: children,
      ),
    );
  }

  // ===========================================================================
  // INTRO CARD
  // ===========================================================================

  Widget _introCard({
    required IconData icon,
    required String title,
    required String subtitle,
  }) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(15),
      decoration: BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
          colors: [
            const Color(0xFF063F42).withOpacity(0.88),
            const Color(0xFF022B2E).withOpacity(0.84),
          ],
        ),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: AppTheme.tealMint.withOpacity(0.17)),
      ),
      child: Row(
        children: [
          Container(
            width: 46,
            height: 46,
            decoration: BoxDecoration(
              color: AppTheme.teal.withOpacity(0.22),
              borderRadius: BorderRadius.circular(14),
            ),
            child: Icon(icon, size: 22, color: AppTheme.tealMint),
          ),

          const SizedBox(width: 12),

          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  title,
                  style: GoogleFonts.dmSans(
                    fontSize: 12,
                    fontWeight: FontWeight.w800,
                    color: Colors.white,
                  ),
                ),

                const SizedBox(height: 4),

                Text(
                  subtitle,
                  style: GoogleFonts.dmSans(
                    fontSize: 9,
                    height: 1.4,
                    color: Colors.white.withOpacity(0.50),
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  // ===========================================================================
  // SECTION HEADING
  // ===========================================================================

  Widget _sectionHeading(String text) {
    return Row(
      children: [
        Container(
          width: 4,
          height: 24,
          decoration: BoxDecoration(
            color: AppTheme.teal,
            borderRadius: BorderRadius.circular(3),
          ),
        ),

        const SizedBox(width: 9),

        Expanded(
          child: Text(
            text,
            style: GoogleFonts.dmSans(
              fontSize: 12.5,
              fontWeight: FontWeight.w800,
              color: Colors.white,
            ),
          ),
        ),
      ],
    );
  }

  // ===========================================================================
  // WHITE INPUT FIELD
  // ===========================================================================

  Widget _field({
    required TextEditingController controller,
    required FocusNode focusNode,
    required String label,
    required String hint,
    required IconData icon,
    TextInputType? keyboardType,
    TextInputAction? textInputAction,
    List<TextInputFormatter>? formatters,
    ValueChanged<String>? onSubmitted,
  }) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        _inputLabel(label),

        const SizedBox(height: 6),

        AnimatedBuilder(
          animation: focusNode,
          builder: (context, child) {
            final focused = focusNode.hasFocus;

            return AnimatedContainer(
              duration: const Duration(milliseconds: 170),
              decoration: BoxDecoration(
                color: Colors.white,
                borderRadius: BorderRadius.circular(16),
                border: Border.all(
                  color: focused
                      ? const Color(0xFF0A6668)
                      : Colors.white.withOpacity(0.65),
                  width: focused ? 1.5 : 0.8,
                ),
                boxShadow: focused
                    ? [
                        BoxShadow(
                          color: AppTheme.teal.withOpacity(0.17),
                          blurRadius: 13,
                        ),
                      ]
                    : null,
              ),
              child: TextField(
                controller: controller,
                focusNode: focusNode,
                keyboardType: keyboardType,
                textInputAction: textInputAction,
                inputFormatters: formatters,
                onSubmitted: onSubmitted,
                autocorrect: false,
                style: GoogleFonts.dmSans(
                  fontSize: 13,
                  fontWeight: FontWeight.w500,
                  color: const Color(0xFF143637),
                ),
                decoration: InputDecoration(
                  hintText: hint,
                  hintStyle: GoogleFonts.dmSans(
                    fontSize: 12,
                    color: const Color(0xFF899899),
                  ),
                  prefixIcon: Icon(
                    icon,
                    size: 19,
                    color: const Color(0xFF0A6668),
                  ),
                  border: InputBorder.none,
                  contentPadding: const EdgeInsets.symmetric(
                    horizontal: 14,
                    vertical: 15,
                  ),
                ),
              ),
            );
          },
        ),
      ],
    );
  }

  // ===========================================================================
  // MULTILINE FIELD
  // ===========================================================================

  Widget _multilineField({
    required TextEditingController controller,
    required FocusNode focusNode,
    required String label,
    required String hint,
    required IconData icon,
  }) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        _inputLabel(label),

        const SizedBox(height: 6),

        AnimatedBuilder(
          animation: focusNode,
          builder: (context, child) {
            final focused = focusNode.hasFocus;

            return AnimatedContainer(
              duration: const Duration(milliseconds: 170),
              decoration: BoxDecoration(
                color: Colors.white,
                borderRadius: BorderRadius.circular(16),
                border: Border.all(
                  color: focused ? const Color(0xFF0A6668) : Colors.white,
                  width: focused ? 1.5 : 0.8,
                ),
              ),
              child: TextField(
                controller: controller,
                focusNode: focusNode,
                minLines: 3,
                maxLines: 5,
                textInputAction: TextInputAction.newline,
                style: GoogleFonts.dmSans(
                  fontSize: 13,
                  height: 1.4,
                  fontWeight: FontWeight.w500,
                  color: const Color(0xFF143637),
                ),
                decoration: InputDecoration(
                  hintText: hint,
                  hintStyle: GoogleFonts.dmSans(
                    fontSize: 12,
                    color: const Color(0xFF899899),
                  ),
                  prefixIcon: Padding(
                    padding: const EdgeInsets.only(bottom: 50),
                    child: Icon(icon, size: 19, color: const Color(0xFF0A6668)),
                  ),
                  border: InputBorder.none,
                  contentPadding: const EdgeInsets.all(15),
                ),
              ),
            );
          },
        ),
      ],
    );
  }

  Widget _inputLabel(String label) {
    return Text(
      label,
      style: GoogleFonts.dmSans(
        fontSize: 10,
        fontWeight: FontWeight.w700,
        color: Colors.white.withOpacity(0.78),
      ),
    );
  }

  // ===========================================================================
  // DROPDOWN
  // ===========================================================================

  Widget _dropdown({
    required String label,
    required String value,
    required IconData icon,
    required List<String> items,
    required ValueChanged<String?> onChanged,
  }) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        _inputLabel(label),

        const SizedBox(height: 6),

        Container(
          height: 51,
          padding: const EdgeInsets.symmetric(horizontal: 13),
          decoration: BoxDecoration(
            color: Colors.white,
            borderRadius: BorderRadius.circular(16),
            border: Border.all(color: Colors.white),
          ),
          child: Row(
            children: [
              Icon(icon, size: 19, color: const Color(0xFF0A6668)),

              const SizedBox(width: 11),

              Expanded(
                child: DropdownButtonHideUnderline(
                  child: DropdownButton<String>(
                    value: value,
                    isExpanded: true,
                    dropdownColor: Colors.white,
                    iconEnabledColor: const Color(0xFF0A6668),
                    borderRadius: BorderRadius.circular(16),
                    items: items.map((item) {
                      return DropdownMenuItem<String>(
                        value: item,
                        child: Text(
                          item,
                          maxLines: 1,
                          overflow: TextOverflow.ellipsis,
                          style: GoogleFonts.dmSans(
                            fontSize: 12,
                            fontWeight: FontWeight.w500,
                            color: const Color(0xFF143637),
                          ),
                        ),
                      );
                    }).toList(),
                    onChanged: onChanged,
                  ),
                ),
              ),
            ],
          ),
        ),
      ],
    );
  }

  // ===========================================================================
  // GLASS CARD
  // ===========================================================================

  Widget _glassCard({required Widget child}) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(15),
      decoration: BoxDecoration(
        color: const Color(0xD909292B),
        borderRadius: BorderRadius.circular(19),
        border: Border.all(color: Colors.white.withOpacity(0.09)),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.11),
            blurRadius: 17,
            offset: const Offset(0, 6),
          ),
        ],
      ),
      child: child,
    );
  }

  // ===========================================================================
  // BOTTOM BAR
  // ===========================================================================

  Widget _buildBottomBar() {
    final keyboardOpen = MediaQuery.viewInsetsOf(context).bottom > 0;

    if (keyboardOpen) {
      return const SizedBox.shrink();
    }

    return Container(
      margin: const EdgeInsets.fromLTRB(14, 4, 14, 12),
      padding: const EdgeInsets.all(8),
      decoration: BoxDecoration(
        color: const Color(0xE8082B2E),
        borderRadius: BorderRadius.circular(21),
        border: Border.all(color: AppTheme.tealMint.withOpacity(0.14)),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.20),
            blurRadius: 20,
            offset: const Offset(0, 7),
          ),
        ],
      ),
      child: SafeArea(
        top: false,
        minimum: EdgeInsets.zero,
        child: Row(
          children: [
            if (_currentStep > 0) ...[
              SizedBox(
                width: 48,
                height: 48,
                child: OutlinedButton(
                  onPressed: _saving ? null : _previousStep,
                  style: OutlinedButton.styleFrom(
                    padding: EdgeInsets.zero,
                    foregroundColor: AppTheme.tealMint,
                    side: BorderSide(
                      color: AppTheme.tealMint.withOpacity(0.38),
                    ),
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(14),
                    ),
                  ),
                  child: const Icon(Icons.arrow_back_rounded, size: 19),
                ),
              ),

              const SizedBox(width: 8),
            ],

            Expanded(
              child: SizedBox(
                height: 48,
                child: ElevatedButton(
                  onPressed: _saving ? null : _nextStep,
                  style: ElevatedButton.styleFrom(
                    backgroundColor: AppTheme.teal,
                    disabledBackgroundColor: AppTheme.teal.withOpacity(0.45),
                    foregroundColor: Colors.white,
                    elevation: 0,
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(14),
                    ),
                  ),
                  child: _saving
                      ? const SizedBox(
                          width: 20,
                          height: 20,
                          child: CircularProgressIndicator(
                            strokeWidth: 2,
                            color: Colors.white,
                          ),
                        )
                      : Row(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            Flexible(
                              child: Text(
                                _currentStep == _totalSteps - 1
                                    ? 'Create Room'
                                    : 'Continue',
                                maxLines: 1,
                                overflow: TextOverflow.ellipsis,
                                style: GoogleFonts.dmSans(
                                  fontSize: 13,
                                  fontWeight: FontWeight.w800,
                                ),
                              ),
                            ),

                            const SizedBox(width: 8),

                            Icon(
                              _currentStep == _totalSteps - 1
                                  ? Icons.check_rounded
                                  : Icons.arrow_forward_rounded,
                              size: 18,
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

  // ===========================================================================
  // ERROR
  // ===========================================================================

  void _showError(String message) {
    ScaffoldMessenger.of(context)
      ..hideCurrentSnackBar()
      ..showSnackBar(
        SnackBar(
          behavior: SnackBarBehavior.floating,
          margin: const EdgeInsets.all(16),
          backgroundColor: const Color(0xFF8B3535),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(15),
          ),
          content: Row(
            children: [
              const Icon(
                Icons.error_outline_rounded,
                color: Colors.white,
                size: 18,
              ),

              const SizedBox(width: 9),

              Expanded(
                child: Text(
                  message,
                  style: GoogleFonts.dmSans(color: Colors.white, fontSize: 11),
                ),
              ),
            ],
          ),
        ),
      );
  }

  // ===========================================================================
  // AMOUNT FORMAT
  // ===========================================================================

  String _formatAmount(int amount) {
    final value = amount.toString();

    final buffer = StringBuffer();

    for (int i = 0; i < value.length; i++) {
      final remaining = value.length - i;

      buffer.write(value[i]);

      if (remaining > 1 && remaining % 3 == 1) {
        buffer.write(',');
      }
    }

    return buffer.toString();
  }
}
